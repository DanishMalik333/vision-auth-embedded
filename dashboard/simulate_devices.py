"""Publish fake Pi + ESP32 messages so the dashboard can be tried without hardware.

    python simulate_devices.py --broker localhost
"""
import argparse
import json
import random
import time

import paho.mqtt.client as mqtt

LABELS = ["Manager", "Supervisor", "CEO", "CFO", "Team Lead", "Engineer", "Intern"]
DEVICES = {
    "rpi-demo": {"platform": "raspberry-pi", "infer": 45.0},
    "esp32-s3-eye": {"platform": "esp32s3", "infer": 120.0},
}


def fake_message(device, info, seq):
    conf = random.uniform(0.5, 1.0)
    diag = ({"cpu_temp_c": random.uniform(48, 62), "cpu_percent": random.uniform(30, 80), "mem_percent": 41.0}
            if info["platform"] == "raspberry-pi"
            else {"free_heap": 180_000, "min_free_heap": 150_000, "free_psram": 6_500_000, "rssi": -58})
    infer = info["infer"] * random.uniform(0.9, 1.1)
    return {
        "device": device, "platform": info["platform"], "seq": seq,
        "label": random.choice(LABELS) if conf >= 0.7 else "Unknown",
        "class_id": 0, "confidence": round(conf, 4), "accepted": conf >= 0.9,
        "latency_ms": {"pre": 3.0, "infer": round(infer, 1), "total": round(infer + 5, 1)},
        "fps": 2.0, "uptime_s": seq // 2, "diag": diag,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic-prefix", default="vision-auth")
    args = parser.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(args.broker, args.port)
    client.loop_start()
    for device in DEVICES:
        client.publish(f"{args.topic_prefix}/{device}/status", "online", qos=1, retain=True)

    seq = 0
    try:
        while True:
            for device, info in DEVICES.items():
                client.publish(f"{args.topic_prefix}/{device}/inference", json.dumps(fake_message(device, info, seq)))
            seq += 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        for device in DEVICES:
            client.publish(f"{args.topic_prefix}/{device}/status", "offline", qos=1, retain=True).wait_for_publish(2)
        client.loop_stop()


if __name__ == "__main__":
    main()
