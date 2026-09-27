set(model_file ${CMAKE_CURRENT_LIST_DIR}/models/model.espdl)
if(NOT CMAKE_BUILD_EARLY_EXPANSION AND NOT EXISTS ${model_file})
    message(FATAL_ERROR
        "Missing ${model_file}\n"
        "Generate it with esp32/quantization/quantize_espdl.py (see README).")
endif()

idf_component_register(
    SRCS app_main.cpp camera.cpp classifier.cpp network.cpp
    INCLUDE_DIRS .
    REQUIRES esp-dl esp32-camera esp_wifi esp_event esp_netif nvs_flash mqtt esp_driver_gpio esp_timer)

# Embed the .espdl model 16-byte aligned in flash rodata (ESP-DL helper).
idf_build_get_property(component_targets __COMPONENT_TARGETS)
if("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)  # read by target_add_aligned_binary_data()
include(${cmake_dir}/utilities.cmake)
target_add_aligned_binary_data(${COMPONENT_LIB} ${model_file} BINARY)
