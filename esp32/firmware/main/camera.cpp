#include "camera.hpp"

#include "esp_log.h"
#include "sdkconfig.h"

static const char *TAG = "camera";

// ESP32-S3-EYE v2.2 pin map (same as esp-who / the esp-bsp board package).
static camera_config_t make_config()
{
    camera_config_t c = {};
    c.pin_pwdn = -1;
    c.pin_reset = -1;
    c.pin_xclk = 15;
    c.pin_sccb_sda = 4;
    c.pin_sccb_scl = 5;
    c.pin_d7 = 16;
    c.pin_d6 = 17;
    c.pin_d5 = 18;
    c.pin_d4 = 12;
    c.pin_d3 = 10;
    c.pin_d2 = 8;
    c.pin_d1 = 9;
    c.pin_d0 = 11;
    c.pin_vsync = 6;
    c.pin_href = 7;
    c.pin_pclk = 13;

    c.xclk_freq_hz = 20000000;
    c.ledc_timer = LEDC_TIMER_0;
    c.ledc_channel = LEDC_CHANNEL_0;

    c.pixel_format = PIXFORMAT_RGB565; // big-endian RGB565, fed to ESP-DL as DL_IMAGE_PIX_TYPE_RGB565BE
    c.frame_size = FRAMESIZE_QVGA;     // 320x240, resized to the model input by the ESP-DL preprocessor
    c.jpeg_quality = 12;
    c.fb_count = 2;
    c.fb_location = CAMERA_FB_IN_PSRAM;
    c.grab_mode = CAMERA_GRAB_LATEST;
    return c;
}

esp_err_t camera_init()
{
    camera_config_t config = make_config();
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_camera_init failed: %s", esp_err_to_name(err));
        return err;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        // Bool Kconfig options that are off are left undefined, hence #ifdef.
#ifdef CONFIG_APP_CAMERA_VFLIP
        s->set_vflip(s, 1);
#endif
#ifdef CONFIG_APP_CAMERA_HMIRROR
        s->set_hmirror(s, 1);
#endif
    }
    ESP_LOGI(TAG, "Camera ready (QVGA RGB565)");
    return ESP_OK;
}
