"""Quantize the ESP32 ONNX model to int8 .espdl with Espressif's esp-ppq (ESP-DL v3).

    python quantize_espdl.py \
        --onnx ../../models/esp32/model.onnx \
        --class-names ../../models/esp32/class_names.json \
        --calib-dir ../../data/split/train \
        --eval-dir ../../data/split/test

Writes, next to the firmware so `idf.py build` picks them up:
    ../firmware/main/models/model.espdl   int8 model embedded into flash
    ../firmware/main/class_names.inc      label table compiled into the firmware
and a quantization report (float vs int8 accuracy) to ../../models/esp32/quant_report.json.

The ONNX model must come from `export_onnx.py --logits --nchw`: NCHW float input in [0, 1], logits output.
"""
import argparse
import json
import os

import numpy as np
import onnx
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.executor.torch import TorchExecutor

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
HERE = os.path.dirname(os.path.abspath(__file__))
FIRMWARE_MAIN = os.path.join(HERE, "..", "firmware", "main")


class FolderDataset(Dataset):
    """Class-per-folder images -> (CHW float tensor in [0, 1], label). Folder order matches Keras (sorted)."""

    def __init__(self, root, img_size, class_names, limit=None):
        self.img_size = img_size
        self.items = []
        for label, cls in enumerate(class_names):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                raise SystemExit(f"Missing class folder: {cls_dir}")
            files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXTS))
            self.items += [(os.path.join(cls_dir, f), label) for f in files]
        if limit:
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(self.items))[:limit]
            self.items = [self.items[i] for i in sorted(idx)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        # Nearest-neighbour resize: the Keras loader default used in training, and what the ESP-DL preprocessor does.
        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size), Image.NEAREST)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return x, label


def onnx_input_shape(path):
    dims = onnx.load(path).graph.input[0].type.tensor_type.shape.dim
    shape = [d.dim_value or 1 for d in dims]
    if len(shape) != 4 or shape[1] != 3:
        raise SystemExit(f"Expected an NCHW [1, 3, H, W] input, got {shape}. Export with --nchw.")
    return shape


def make_setting(args):
    setting = QuantizationSettingFactory.espdl_setting()
    if args.equalization:
        # Layerwise weight equalization (Nagel et al. 2019), as in Espressif's MobileNetV2 tutorial.
        # Without it, per-tensor int8 scales wipe out the small-range channels of the depthwise
        # blocks and accuracy collapses. Requires ReLU (not ReLU6) activations, see train.py.
        setting.equalization = True
        eq = setting.equalization_setting
        eq.iterations = 6
        eq.value_threshold = 0.5
        eq.opt_level = 2
        eq.interested_layers = None
    return setting


def quantize(args, input_shape, calib_loader):
    common = dict(
        onnx_import_file=args.onnx,
        espdl_export_file=args.output,
        calib_dataloader=calib_loader,
        calib_steps=args.calib_steps,
        input_shape=input_shape,
        target=args.target,
        collate_fn=lambda batch: batch[0].to(args.device),
        device=args.device,
        error_report=True,
        skip_export=False,
        export_test_values=True,  # lets the firmware self-test with model->test()
        setting=make_setting(args),
        verbose=1,
    )
    try:
        return espdl_quantize_onnx(quant_type=args.quant_type, **common)
    except TypeError:
        # esp-ppq releases before `quant_type` existed only take a uniform bit width.
        return espdl_quantize_onnx(num_of_bits=16 if args.quant_type == "w16a16" else 8, **common)


def evaluate(args, graph, eval_set):
    import onnxruntime as ort

    loader = DataLoader(eval_set, batch_size=1, shuffle=False)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    executor = TorchExecutor(graph=graph, device=args.device)

    float_ok = int8_ok = agree = 0
    for x, y in loader:
        f = int(np.argmax(sess.run(None, {in_name: x.numpy()})[0]))
        q = int(torch.argmax(executor(x.to(args.device))[0]).item())
        float_ok += f == int(y)
        int8_ok += q == int(y)
        agree += f == q
    n = len(eval_set)
    return {"samples": n, "float_accuracy": float_ok / n, "int8_accuracy": int8_ok / n, "float_int8_agreement": agree / n}


def write_class_names(class_names, onnx_shape):
    path = os.path.join(FIRMWARE_MAIN, "class_names.inc")
    with open(path, "w", encoding="utf-8") as f:
        f.write("// Generated by esp32/quantization/quantize_espdl.py - do not edit.\n")
        f.write(f"// Model input: {onnx_shape[2]}x{onnx_shape[3]} RGB, {len(class_names)} classes.\n")
        for name in class_names:
            f.write(f"    {json.dumps(name)},\n")
    print(f"[INFO] Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", default="../../models/esp32/model.onnx")
    parser.add_argument("--class-names", default="../../models/esp32/class_names.json")
    parser.add_argument("--calib-dir", default="../../data/split/train")
    parser.add_argument("--eval-dir", default="../../data/split/test", help="Set to '' to skip evaluation")
    parser.add_argument("--output", default=os.path.join(FIRMWARE_MAIN, "models", "model.espdl"))
    parser.add_argument("--report", default="../../models/esp32/quant_report.json")
    parser.add_argument("--target", default="esp32s3", choices=["esp32s3", "esp32p4", "c"])
    parser.add_argument("--quant-type", default="w8a8", choices=["w8a8", "w8a16", "w16a16"],
                        help="w8a8 = full int8 (default); w8a16 keeps int8 weights with int16 activations")
    parser.add_argument("--no-equalization", dest="equalization", action="store_false")
    parser.add_argument("--calib-images", type=int, default=512)
    parser.add_argument("--calib-steps", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.class_names) as f:
        class_names = json.load(f)
    shape = onnx_input_shape(args.onnx)
    img_size = shape[2]
    if args.equalization and any(n.op_type == "Clip" for n in onnx.load(args.onnx).graph.node):
        print("[WARN] Model contains Clip (ReLU6) ops; equalization works best with ReLU. "
              "Train with `train.py --target esp32`.")

    calib_set = FolderDataset(args.calib_dir, img_size, class_names, limit=args.calib_images)
    # shuffle must stay False: esp-ppq iterates the loader several times when measuring error.
    calib_loader = DataLoader(calib_set, batch_size=32, shuffle=False)
    print(f"[INFO] Calibrating on {len(calib_set)} images at {img_size}x{img_size} for {args.target}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    graph = quantize(args, shape, calib_loader)
    print(f"[INFO] Wrote {args.output} ({os.path.getsize(args.output) / 1024:.0f} KiB)")

    write_class_names(class_names, shape)

    if args.eval_dir:
        report = evaluate(args, graph, FolderDataset(args.eval_dir, img_size, class_names))
        report.update(target=args.target, quant_type=args.quant_type, equalization=args.equalization,
                      input_size=img_size,
                      espdl_kib=round(os.path.getsize(args.output) / 1024, 1))
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[RESULT] {json.dumps(report)}")


if __name__ == "__main__":
    main()
