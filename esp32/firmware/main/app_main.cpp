// Vision-based authentication on the ESP32-S3-EYE.
//
// Loop: grab a QVGA RGB565 frame -> centre-crop + resize + quantize (ESP-DL preprocessor)
//       -> int8 MobileNetV2 on the CPU -> softmax -> thresholds -> JSON over MQTT.
#include <algorithm>
#include <cstdio>

#include "camera.hpp"
#include "classifier.hpp"
#include "driver/gpio.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "network.hpp"
#include "nvs_flash.h"
#include "sdkconfig.h"

static const char *TAG = "app";

static constexpr float UNKNOWN_THRESHOLD = CONFIG_APP_UNKNOWN_THRESHOLD_PCT / 100.0f;
static constexpr float ACCEPT_THRESHOLD = CONFIG_APP_ACCEPT_THRESHOLD_PCT / 100.0f;

static void init_nvs()
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
}

#if CONFIG_APP_TRIGGER_BUTTON
static void init_button()
{
    gpio_config_t io = {};
    io.pin_bit_mask = 1ULL << CONFIG_APP_BUTTON_GPIO;
    io.mode = GPIO_MODE_INPUT;
    io.pull_up_en = GPIO_PULLUP_ENABLE;
    ESP_ERROR_CHECK(gpio_config(&io));
}

// Blocks until the (active-low) button is pressed and released.
static void wait_for_button()
{
    while (gpio_get_level((gpio_num_t)CONFIG_APP_BUTTON_GPIO) != 0) {
        vTaskDelay(pdMS_TO_TICKS(20));
    }
    while (gpio_get_level((gpio_num_t)CONFIG_APP_BUTTON_GPIO) == 0) {
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}
#endif

static int format_payload(char *buf, size_t size, uint32_t seq, const Prediction &p, const char *label, bool accepted,
                          float total_ms, float fps)
{
    return snprintf(buf, size,
                    "{\"device\":\"%s\",\"platform\":\"esp32s3\",\"seq\":%lu,"
                    "\"label\":\"%s\",\"class_id\":%d,\"confidence\":%.4f,\"accepted\":%s,"
                    "\"latency_ms\":{\"pre\":%.1f,\"infer\":%.1f,\"post\":%.2f,\"total\":%.1f},"
                    "\"fps\":%.2f,\"uptime_s\":%lld,"
                    "\"diag\":{\"free_heap\":%u,\"min_free_heap\":%u,\"free_psram\":%u,\"rssi\":%d}}",
                    CONFIG_APP_DEVICE_ID, (unsigned long)seq, label, p.class_id, p.confidence,
                    accepted ? "true" : "false", p.preprocess_us / 1000.0f, p.inference_us / 1000.0f,
                    p.postprocess_us / 1000.0f, total_ms, fps, (long long)(esp_timer_get_time() / 1000000),
                    (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                    (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
                    (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), wifi_rssi());
}

extern "C" void app_main(void)
{
    init_nvs();
    ESP_ERROR_CHECK(camera_init());

    static Classifier classifier;
    ESP_ERROR_CHECK(classifier.init());

    ESP_ERROR_CHECK(wifi_connect());
    ESP_ERROR_CHECK(mqtt_start());

#if CONFIG_APP_TRIGGER_BUTTON
    init_button();
    ESP_LOGI(TAG, "Ready - press the BOOT button to authenticate");
#else
    ESP_LOGI(TAG, "Ready - running continuously every %d ms", CONFIG_APP_INFERENCE_INTERVAL_MS);
#endif

    static char payload[512];
    uint32_t seq = 0;
    int64_t last_frame_us = esp_timer_get_time();

    while (true) {
#if CONFIG_APP_TRIGGER_BUTTON
        wait_for_button();
#endif
        int64_t start_us = esp_timer_get_time();
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGW(TAG, "Camera capture failed");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        dl::image::img_t img = {
            .data = fb->buf,
            .width = (uint16_t)fb->width,
            .height = (uint16_t)fb->height,
            .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE,
        };
        // Whole frame resized (squashed) to the model input, matching how the training images were captured.
        Prediction p = classifier.classify(img, {});
        esp_camera_fb_return(fb);

        int64_t now_us = esp_timer_get_time();
        float total_ms = (now_us - start_us) / 1000.0f;
        float fps = 1e6f / std::max<int64_t>(now_us - last_frame_us, 1);
        last_frame_us = now_us;

        const char *label = p.confidence < UNKNOWN_THRESHOLD ? "Unknown" : p.label;
        bool accepted = p.confidence >= ACCEPT_THRESHOLD;

        ESP_LOGI(TAG, "%s (%.2f)%s  pre %.1f ms | infer %.1f ms | total %.1f ms", label, p.confidence,
                 accepted ? " ACCEPTED" : "", p.preprocess_us / 1000.0f, p.inference_us / 1000.0f, total_ms);

        int len = format_payload(payload, sizeof(payload), seq++, p, label, accepted, total_ms, fps);
        if (len > 0 && len < (int)sizeof(payload)) {
            mqtt_publish_inference(payload, len);
        }

#if !CONFIG_APP_TRIGGER_BUTTON
        vTaskDelay(pdMS_TO_TICKS(CONFIG_APP_INFERENCE_INTERVAL_MS));
#endif
    }
}
