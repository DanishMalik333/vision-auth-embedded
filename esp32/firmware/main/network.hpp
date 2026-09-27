#pragma once

#include "esp_err.h"

// Connect to Wi-Fi (blocks until an IP is obtained; reconnects automatically afterwards).
esp_err_t wifi_connect();

// Wi-Fi signal strength of the current AP in dBm, or 0 when not connected.
int wifi_rssi();

// Start the MQTT client. Publishes "online" (retained) to <prefix>/<id>/status on connect
// and registers "offline" as the last will.
esp_err_t mqtt_start();

bool mqtt_is_connected();

// Publish a JSON payload to <prefix>/<id>/inference (QoS 0).
void mqtt_publish_inference(const char *json, int len);
