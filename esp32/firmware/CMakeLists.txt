# ESP32-S3-EYE firmware: camera -> int8 MobileNetV2 (ESP-DL) -> MQTT
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(vision_auth_esp32)
