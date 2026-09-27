"""Real-time authentication on a Raspberry Pi: webcam -> MobileNetV2 (ONNX Runtime, CPU) -> MQTT.

    python inference.py --broker 192.168.1.10
    python inference.py --broker 192.168.1.10 --trigger button --button-pin 17
    python inference.py --broker localhost --show          # also runs on a laptop for testing

Publishes JSON to <prefix>/<device-id>/inference and "online"/"offline" (retained) to
<prefix>/<device-id>/status. The payload format is shared with the ESP32 firmware (see docs/mqtt.md).
"""
import argparse
import json
import os
import socket
import time

import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt

try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):
    GPIO = None

try:
    import psutil
except ImportError:
    psutil = None

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(HERE, "..", "models", "pi", "model.onnx"))
    p.add_argument("--labels", default=os.path.join(HERE, "..", "models", "pi", "class_names.json"))
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--device-id", default=f"rpi-{socket.gethostname()}")
    p.add_argument("--topic-prefix", default="vision-auth")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--interval", type=float, default=0.5, help="Seconds between inferences (continuous mode)")
    p.add_argument("--trigger", choices=["continuous", "button"], default="continuous")
    p.add_argument("--button-pin", type=int, default=17, help="BCM pin, pulled down, active high")
    p.add_argument("--unknown-threshold", type=float, default=0.7)
    p.add_argument("--accept-threshold", type=float, default=0.9)
    p.add_argument("--threads", type=int, default=4, help="ONNX Runtime intra-op threads")
    p.add_argument("--show", action="store_true", help="Show the camera feed with the prediction")
    return p.parse_args()


class Classifier:
    def __init__(self, model_path, labels_path, threads):
        with open(labels_path) as f:
            self.labels = json.load(f)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Accept both NHWC (tf2onnx default) and NCHW exports.
        self.nchw = inp.shape[1] == 3
        self.height, self.width = (inp.shape[2], inp.shape[3]) if self.nchw else (inp.shape[1], inp.shape[2])

    def preprocess(self, frame_bgr):
        # Full frame squashed to the model size, exactly like capture_dataset.py stored the training images.
        img = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        # The model was trained on RGB images (Keras loader) scaled to [0, 1]; OpenCV delivers BGR.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.nchw:
            img = img.transpose(2, 0, 1)
        return img[np.newaxis]

    def classify(self, frame_bgr):
        t0 = time.perf_counter()
        x = self.preprocess(frame_bgr)
        t1 = time.perf_counter()
        probs = self.session.run(None, {self.input_name: x})[0][0]
        t2 = time.perf_counter()
        class_id = int(np.argmax(probs))
        return class_id, float(probs[class_id]), (t1 - t0) * 1e3, (t2 - t1) * 1e3


def read_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except OSError:
        return None


def diagnostics():
    diag = {"cpu_temp_c": read_cpu_temp()}
    if psutil:
        diag["cpu_percent"] = psutil.cpu_percent()
        diag["mem_percent"] = psutil.virtual_memory().percent
    return diag


def make_mqtt(args):
    status_topic = f"{args.topic_prefix}/{args.device_id}/status"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=args.device_id)
    client.will_set(status_topic, "offline", qos=1, retain=True)

    def on_connect(c, userdata, flags, reason_code, properties):
        print(f"[MQTT] connected to {args.broker}:{args.port} ({reason_code})")
        c.publish(status_topic, "online", qos=1, retain=True)

    client.on_connect = on_connect
    client.connect_async(args.broker, args.port, keepalive=15)
    client.loop_start()  # background thread, reconnects automatically
    return client, status_topic


def wait_for_button(pin):
    while GPIO.input(pin) == GPIO.LOW:
        time.sleep(0.02)
    while GPIO.input(pin) == GPIO.HIGH:
        time.sleep(0.02)


def main():
    args = parse_args()
    clf = Classifier(args.model, args.labels, args.threads)
    print(f"[INFO] Model {os.path.basename(args.model)}: {clf.width}x{clf.height}, {len(clf.labels)} classes")

    if args.trigger == "button":
        if GPIO is None:
            raise SystemExit("RPi.GPIO is not available - use --trigger continuous on this machine")
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(args.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    client, status_topic = make_mqtt(args)
    inference_topic = f"{args.topic_prefix}/{args.device_id}/inference"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    print("[INFO] Ready" + (" - press the button to authenticate" if args.trigger == "button" else ""))
    seq, last_frame, start_time = 0, time.perf_counter(), time.time()
    try:
        while True:
            if args.trigger == "button":
                wait_for_button(args.button_pin)
            t_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Camera read failed")
                time.sleep(0.1)
                continue

            class_id, conf, pre_ms, infer_ms = clf.classify(frame)
            label = clf.labels[class_id] if conf >= args.unknown_threshold else "Unknown"
            accepted = conf >= args.accept_threshold

            now = time.perf_counter()
            total_ms = (now - t_start) * 1e3
            fps = 1.0 / max(now - last_frame, 1e-6)
            last_frame = now

            payload = {
                "device": args.device_id,
                "platform": "raspberry-pi",
                "seq": seq,
                "label": label,
                "class_id": class_id,
                "confidence": round(conf, 4),
                "accepted": accepted,
                "latency_ms": {"pre": round(pre_ms, 1), "infer": round(infer_ms, 1), "total": round(total_ms, 1)},
                "fps": round(fps, 2),
                "uptime_s": int(time.time() - start_time),
                "diag": diagnostics(),
            }
            client.publish(inference_topic, json.dumps(payload))
            seq += 1
            print(f"{label} ({conf:.2f}){' ACCEPTED' if accepted else ''}  infer {infer_ms:.1f} ms")

            if args.show:
                cv2.putText(frame, f"{label} {conf:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if accepted else (0, 200, 255), 2)
                cv2.imshow("vision-auth", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.trigger == "continuous":
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        client.publish(status_topic, "offline", qos=1, retain=True).wait_for_publish(timeout=2)
        client.loop_stop()
        client.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        if GPIO is not None and args.trigger == "button":
            GPIO.cleanup()


if __name__ == "__main__":
    main()
