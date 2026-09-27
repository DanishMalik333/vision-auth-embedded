"""Train a MobileNetV2 classifier for the Raspberry Pi or the ESP32-S3.

Two presets share one pipeline (ImageNet-pretrained backbone -> new head -> fine-tune last layers):

    pi     MobileNetV2 alpha=1.0,  224x224 input  (ONNX Runtime on the Pi CPU)
    esp32  MobileNetV2 alpha=0.35,  96x96 input   (int8 ESP-DL on the ESP32-S3)

    python train.py --target pi    --data ../data/split
    python train.py --target esp32 --data ../data/split

Outputs go to ../models/<target>/: model.keras, class_names.json and metrics.json.
The last Dense layer is named "logits" and is followed by a separate softmax layer, so
export_onnx.py --logits can strip the softmax for the microcontroller build.
"""
import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PRESETS = {
    "pi": {"alpha": 1.0, "img_size": 224, "relu6": True},
    # ReLU instead of ReLU6 lets esp-ppq apply layerwise equalization, which is what makes the
    # per-tensor power-of-two int8 quantization of ESP-DL accurate for MobileNetV2.
    "esp32": {"alpha": 0.35, "img_size": 96, "relu6": False},
}


def replace_relu6(model):
    """Clone a Keras model with every ReLU(max_value=6) swapped for a plain ReLU, keeping the weights."""

    def clone(layer):
        if isinstance(layer, tf.keras.layers.ReLU) and layer.max_value is not None:
            return tf.keras.layers.ReLU(name=layer.name)
        return layer.__class__.from_config(layer.get_config())

    new = tf.keras.models.clone_model(model, clone_function=clone)
    new.set_weights(model.get_weights())
    return new


def build_model(num_classes, img_size, alpha, relu6=True):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), alpha=alpha, include_top=False, weights="imagenet"
    )
    if not relu6:
        base = replace_relu6(base)
    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(num_classes, name="logits")(x)
    out = tf.keras.layers.Softmax(name="probs")(x)
    return tf.keras.Model(base.input, out), base


def make_generators(data_dir, img_size, batch_size):
    # Inputs are RGB scaled to [0, 1]; every deployment target reproduces exactly this.
    train_gen = ImageDataGenerator(rescale=1 / 255.0, horizontal_flip=True, rotation_range=20, zoom_range=0.2)
    plain_gen = ImageDataGenerator(rescale=1 / 255.0)
    common = dict(target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical")

    train = train_gen.flow_from_directory(os.path.join(data_dir, "train"), **common)
    val = plain_gen.flow_from_directory(os.path.join(data_dir, "val"), **common)
    test = plain_gen.flow_from_directory(os.path.join(data_dir, "test"), shuffle=False, **common)
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=PRESETS, default="pi")
    parser.add_argument("--data", default="../data/split", help="Folder with train/ val/ test/ sub-folders")
    parser.add_argument("--img-size", type=int, help="Override the preset input size")
    parser.add_argument("--alpha", type=float, help="Override the preset MobileNetV2 width multiplier")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", help="Defaults to ../models/<target>")
    args = parser.parse_args()

    preset = PRESETS[args.target]
    img_size = args.img_size or preset["img_size"]
    alpha = args.alpha or preset["alpha"]
    out_dir = args.out_dir or os.path.join("..", "models", args.target)
    os.makedirs(out_dir, exist_ok=True)

    train, val, test = make_generators(args.data, img_size, args.batch_size)
    class_names = list(train.class_indices.keys())
    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)
    print(f"[INFO] Classes: {class_names}")

    model, base = build_model(len(class_names), img_size, alpha, preset["relu6"])
    model.compile(optimizer=tf.keras.optimizers.Adam(8e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    print(f"\n[INFO] Training head (MobileNetV2 alpha={alpha}, {img_size}x{img_size})")
    model.fit(train, validation_data=val, epochs=args.epochs)

    print(f"\n[INFO] Fine-tuning last {args.fine_tune_layers} backbone layers")
    base.trainable = True
    for layer in base.layers[:-args.fine_tune_layers]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.fine_tune_epochs)

    loss, acc = model.evaluate(test)
    print(f"\n[TEST] accuracy={acc:.4f} loss={loss:.4f}")

    model_path = os.path.join(out_dir, "model.keras")
    model.save(model_path)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"target": args.target, "alpha": alpha, "img_size": img_size, "relu6": preset["relu6"],
                   "params": model.count_params(), "test_accuracy": acc, "test_loss": loss}, f, indent=2)
    print(f"[INFO] Saved {model_path}")


if __name__ == "__main__":
    main()
