"""Convert a trained Keras model to ONNX with tf2onnx.

Raspberry Pi (keeps softmax, NHWC input - what inference.py expects):
    python export_onnx.py --model ../models/pi/model.keras --output ../models/pi/model.onnx

ESP32 (drops softmax and uses an NCHW input, which is the layout esp-ppq calibrates with):
    python export_onnx.py --model ../models/esp32/model.keras --output ../models/esp32/model.onnx --logits --nchw
"""
import argparse

import numpy as np
import onnx
import tensorflow as tf
import tf2onnx


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help=".keras or .h5 model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--logits", action="store_true",
                        help="Export the 'logits' layer output (no softmax); softmax runs on-device in float")
    parser.add_argument("--nchw", action="store_true", help="Make the ONNX input NCHW instead of NHWC")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    if args.logits:
        try:
            model = tf.keras.Model(model.input, model.get_layer("logits").output)
        except ValueError:
            raise SystemExit("No layer named 'logits' - retrain with train.py or export without --logits")

    _, h, w, c = model.input_shape
    spec = (tf.TensorSpec((1, h, w, c), tf.float32, name="input"),)
    extra = {"inputs_as_nchw": ["input"]} if args.nchw else {}
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=args.opset, **extra)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, args.output)

    # Sanity check: ONNX Runtime must reproduce the Keras output.
    try:
        import onnxruntime as ort

        x = np.random.rand(1, h, w, c).astype(np.float32)
        ref = model.predict(x, verbose=0)
        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        feed = x.transpose(0, 3, 1, 2) if args.nchw else x
        got = sess.run(None, {sess.get_inputs()[0].name: feed})[0]
        print(f"[CHECK] max |keras - onnx| = {np.abs(ref - got).max():.2e}")
    except ImportError:
        print("[WARN] onnxruntime not installed, skipped numerical check")

    layout = "NCHW" if args.nchw else "NHWC"
    print(f"[INFO] Saved {args.output} (input {layout}, {'logits' if args.logits else 'softmax'} output)")


if __name__ == "__main__":
    main()
