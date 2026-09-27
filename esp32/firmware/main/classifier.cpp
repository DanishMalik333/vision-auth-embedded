#include "classifier.hpp"

#include <cmath>

#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "classifier";

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

static const char *const CLASS_NAMES[] = {
#include "class_names.inc"
};
static constexpr int NUM_LABELS = sizeof(CLASS_NAMES) / sizeof(CLASS_NAMES[0]);

esp_err_t Classifier::init()
{
    m_model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    // Test vectors exported by esp-ppq (export_test_values=True) catch conversion mistakes on first boot.
    // Must run before minimize(), which frees them.
    if (m_model->test() != ESP_OK) {
        ESP_LOGW(TAG, "Model self-test against esp-ppq reference outputs FAILED");
    }
    m_model->minimize();

    // Training scaled RGB to [0, 1]: (x - 0) / 255. The preprocessor resizes, normalizes and quantizes
    // straight into the model's int8 input tensor (NHWC).
    m_preprocessor = new dl::image::ImagePreprocessor(m_model, {0, 0, 0}, {255, 255, 255});

    const std::vector<int> &in_shape = m_model->get_input()->get_shape();
    m_input_h = in_shape[1];
    m_input_w = in_shape[2];

    m_output = m_model->get_output();
    m_num_classes = m_output->get_shape().back();
    m_logits = new dl::TensorBase(m_output->get_shape(), nullptr, 0, dl::DATA_TYPE_FLOAT);
    m_probs.resize(m_num_classes);

    if (m_num_classes != NUM_LABELS) {
        ESP_LOGE(TAG, "Model has %d outputs but class_names.inc lists %d labels", m_num_classes, NUM_LABELS);
        return ESP_ERR_INVALID_SIZE;
    }
    ESP_LOGI(TAG, "Model loaded: input %dx%d, %d classes, output %s", m_input_w, m_input_h, m_num_classes,
             m_output->get_dtype_string());
    return ESP_OK;
}

Prediction Classifier::classify(const dl::image::img_t &img, const std::vector<int> &crop_area)
{
    Prediction p = {};

    int64_t t0 = esp_timer_get_time();
    m_preprocessor->preprocess(img, crop_area);
    int64_t t1 = esp_timer_get_time();
    m_model->run();
    int64_t t2 = esp_timer_get_time();

    // The model outputs logits (softmax stripped before quantization for better int8 resolution),
    // so dequantize and apply softmax in float here.
    m_logits->assign(m_output);
    const float *logits = (const float *)m_logits->data;

    float max_logit = logits[0];
    for (int i = 1; i < m_num_classes; i++) {
        max_logit = std::fmax(max_logit, logits[i]);
    }
    float sum = 0.f;
    for (int i = 0; i < m_num_classes; i++) {
        m_probs[i] = std::exp(logits[i] - max_logit);
        sum += m_probs[i];
    }
    int best = 0;
    for (int i = 0; i < m_num_classes; i++) {
        m_probs[i] /= sum;
        if (m_probs[i] > m_probs[best]) {
            best = i;
        }
    }
    int64_t t3 = esp_timer_get_time();

    p.class_id = best;
    p.label = CLASS_NAMES[best];
    p.confidence = m_probs[best];
    p.preprocess_us = t1 - t0;
    p.inference_us = t2 - t1;
    p.postprocess_us = t3 - t2;
    return p;
}
