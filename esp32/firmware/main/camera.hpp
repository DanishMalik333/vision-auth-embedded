#pragma once

#include "esp_camera.h"
#include "esp_err.h"

// ESP32-S3-EYE (OV2640) camera: QVGA RGB565 frames in PSRAM, always the latest frame.
esp_err_t camera_init();
