#pragma once

#include <cstdint>
#include <vector>

#include "dl_image_define.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"
#include "esp_err.h"

struct Prediction {
    int class_id;
    const char *label;
    float confidence;
    int64_t preprocess_us;
    int64_t inference_us;
    int64_t postprocess_us;
};

// int8 MobileNetV2 embedded in flash (model.espdl), run on the CPU with ESP-DL.
class Classifier {
public:
    esp_err_t init();
    // crop_area = {x0, y0, x1, y1} in source pixels, or empty for the full frame.
    Prediction classify(const dl::image::img_t &img, const std::vector<int> &crop_area);

    int input_width() const { return m_input_w; }
    int input_height() const { return m_input_h; }
    int num_classes() const { return m_num_classes; }

private:
    dl::Model *m_model = nullptr;
    dl::image::ImagePreprocessor *m_preprocessor = nullptr;
    dl::TensorBase *m_output = nullptr; // model output as int8/int16
    dl::TensorBase *m_logits = nullptr; // dequantized float copy
    std::vector<float> m_probs;
    int m_input_w = 0;
    int m_input_h = 0;
    int m_num_classes = 0;
};
