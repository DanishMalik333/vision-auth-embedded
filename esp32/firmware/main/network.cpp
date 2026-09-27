#include "network.hpp"

#include <cstring>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "mqtt_client.h"
#include "sdkconfig.h"

static const char *TAG = "network";

static EventGroupHandle_t s_wifi_events;
static constexpr EventBits_t WIFI_CONNECTED_BIT = BIT0;

static esp_mqtt_client_handle_t s_mqtt = nullptr;
static volatile bool s_mqtt_connected = false;
static char s_status_topic[128];
static char s_inference_topic[128];

// ---------------------------------------------------------------- Wi-Fi

static void wifi_event_handler(void *, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_wifi_events, WIFI_CONNECTED_BIT);
        ESP_LOGW(TAG, "Wi-Fi disconnected, retrying");
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        auto *event = static_cast<ip_event_got_ip_t *>(data);
        ESP_LOGI(TAG, "Got IP " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_wifi_events, WIFI_CONNECTED_BIT);
    }
}

esp_err_t wifi_connect()
{
    s_wifi_events = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, nullptr, nullptr));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, nullptr, nullptr));

    wifi_config_t cfg = {};
    strncpy((char *)cfg.sta.ssid, CONFIG_APP_WIFI_SSID, sizeof(cfg.sta.ssid) - 1);
    strncpy((char *)cfg.sta.password, CONFIG_APP_WIFI_PASSWORD, sizeof(cfg.sta.password) - 1);
    cfg.sta.threshold.authmode = strlen(CONFIG_APP_WIFI_PASSWORD) ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &cfg));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to '%s'...", CONFIG_APP_WIFI_SSID);
    xEventGroupWaitBits(s_wifi_events, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
    return ESP_OK;
}

int wifi_rssi()
{
    wifi_ap_record_t ap;
    return esp_wifi_sta_get_ap_info(&ap) == ESP_OK ? ap.rssi : 0;
}

// ---------------------------------------------------------------- MQTT

static void mqtt_event_handler(void *, esp_event_base_t, int32_t event_id, void *)
{
    switch ((esp_mqtt_event_id_t)event_id) {
    case MQTT_EVENT_CONNECTED:
        s_mqtt_connected = true;
        esp_mqtt_client_publish(s_mqtt, s_status_topic, "online", 0, 1, 1);
        ESP_LOGI(TAG, "MQTT connected to %s", CONFIG_APP_MQTT_BROKER_URI);
        break;
    case MQTT_EVENT_DISCONNECTED:
        s_mqtt_connected = false;
        ESP_LOGW(TAG, "MQTT disconnected (client retries automatically)");
        break;
    case MQTT_EVENT_ERROR:
        ESP_LOGW(TAG, "MQTT error");
        break;
    default:
        break;
    }
}

esp_err_t mqtt_start()
{
    snprintf(s_status_topic, sizeof(s_status_topic), "%s/%s/status", CONFIG_APP_TOPIC_PREFIX, CONFIG_APP_DEVICE_ID);
    snprintf(s_inference_topic, sizeof(s_inference_topic), "%s/%s/inference", CONFIG_APP_TOPIC_PREFIX,
             CONFIG_APP_DEVICE_ID);

    esp_mqtt_client_config_t cfg = {};
    cfg.broker.address.uri = CONFIG_APP_MQTT_BROKER_URI;
    cfg.credentials.client_id = CONFIG_APP_DEVICE_ID;
    cfg.session.last_will.topic = s_status_topic;
    cfg.session.last_will.msg = "offline";
    cfg.session.last_will.qos = 1;
    cfg.session.last_will.retain = 1;
    cfg.session.keepalive = 15;

    s_mqtt = esp_mqtt_client_init(&cfg);
    if (!s_mqtt) {
        return ESP_FAIL;
    }
    esp_mqtt_client_register_event(s_mqtt, MQTT_EVENT_ANY, mqtt_event_handler, nullptr);
    return esp_mqtt_client_start(s_mqtt);
}

bool mqtt_is_connected()
{
    return s_mqtt_connected;
}

void mqtt_publish_inference(const char *json, int len)
{
    if (s_mqtt_connected) {
        esp_mqtt_client_publish(s_mqtt, s_inference_topic, json, len, 0, 0);
    }
}
