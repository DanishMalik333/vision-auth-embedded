"""Monitoring dashboard: one live card per edge device (Raspberry Pi, ESP32, ...).

    python dashboard.py --broker localhost

Subscribes to <prefix>/+/inference and <prefix>/+/status. Devices appear automatically the first
time they publish. Each card shows the latest accepted identity, confidence, latency breakdown,
throughput and device diagnostics (heap/PSRAM/RSSI on the ESP32, CPU temperature/load on the Pi).
"""
import argparse
import json
import queue
import time
import tkinter as tk

import paho.mqtt.client as mqtt

BG = "#1e1e1e"
CARD = "#2a2a2a"
FG = "#e8e8e8"
MUTED = "#9a9a9a"
GREEN = "#00d68f"
AMBER = "#ffaa00"
RED = "#ff5c5c"
STALE_AFTER_S = 3.0


def fmt_bytes(n):
    return f"{n / 1024:.0f} KiB" if n is not None else "-"


class DeviceCard:
    def __init__(self, parent, device_id):
        self.frame = tk.Frame(parent, bg=CARD, padx=16, pady=12)
        self.last_seen = 0.0
        self.online = None

        header = tk.Frame(self.frame, bg=CARD)
        header.pack(fill="x")
        self.dot = tk.Label(header, text="●", fg=MUTED, bg=CARD, font=("Segoe UI", 14))
        self.dot.pack(side="left")
        tk.Label(header, text=device_id, fg=FG, bg=CARD, font=("Segoe UI", 13, "bold")).pack(side="left", padx=6)
        self.platform = tk.Label(header, text="", fg=MUTED, bg=CARD, font=("Segoe UI", 10))
        self.platform.pack(side="right")

        self.label = tk.Label(self.frame, text="Waiting...", fg=MUTED, bg=CARD, font=("Segoe UI", 28, "bold"))
        self.label.pack(pady=(10, 0))
        self.conf = tk.Label(self.frame, text="", fg=MUTED, bg=CARD, font=("Segoe UI", 12))
        self.conf.pack()

        self.bar = tk.Canvas(self.frame, height=8, bg="#3a3a3a", highlightthickness=0)
        self.bar.pack(fill="x", pady=8)

        self.latency = tk.Label(self.frame, text="", fg=FG, bg=CARD, font=("Consolas", 10), justify="left")
        self.latency.pack(anchor="w")
        self.diag = tk.Label(self.frame, text="", fg=MUTED, bg=CARD, font=("Consolas", 10), justify="left")
        self.diag.pack(anchor="w")

    def set_status(self, online):
        self.online = online
        self.dot.config(fg=GREEN if online else RED)

    def update(self, msg):
        self.last_seen = time.time()
        if self.online is None:
            self.set_status(True)
        self.platform.config(text=msg.get("platform", ""))

        conf = float(msg.get("confidence", 0.0))
        label = msg.get("label", "?")
        accepted = msg.get("accepted", False)
        color = GREEN if accepted else (AMBER if label != "Unknown" else MUTED)
        self.label.config(text=label, fg=color)
        self.conf.config(text=f"confidence {conf:.1%}" + ("  ✓ access granted" if accepted else ""))

        self.bar.update_idletasks()
        width = self.bar.winfo_width()
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, int(width * conf), 8, fill=color, width=0)

        lat = msg.get("latency_ms", {})
        self.latency.config(
            text=f"pre {lat.get('pre', 0):6.1f} ms | infer {lat.get('infer', 0):7.1f} ms | "
                 f"total {lat.get('total', 0):7.1f} ms\n{msg.get('fps', 0):.2f} fps | seq {msg.get('seq', 0)} | "
                 f"up {msg.get('uptime_s', 0)} s")

        d = msg.get("diag", {})
        parts = []
        if "free_heap" in d:
            parts.append(f"heap {fmt_bytes(d['free_heap'])} (min {fmt_bytes(d.get('min_free_heap'))})")
        if "free_psram" in d:
            parts.append(f"psram {fmt_bytes(d['free_psram'])}")
        if d.get("rssi"):
            parts.append(f"rssi {d['rssi']} dBm")
        if d.get("cpu_temp_c") is not None:
            parts.append(f"cpu {d['cpu_temp_c']:.1f} °C")
        if "cpu_percent" in d:
            parts.append(f"load {d['cpu_percent']:.0f}%")
        if "mem_percent" in d:
            parts.append(f"mem {d['mem_percent']:.0f}%")
        self.diag.config(text=" | ".join(parts))

    def check_stale(self):
        if self.last_seen and time.time() - self.last_seen > STALE_AFTER_S:
            self.label.config(text="No detection", fg=MUTED)
            self.conf.config(text=f"last message {time.time() - self.last_seen:.0f} s ago")
            self.bar.delete("all")


class Dashboard:
    def __init__(self, root, args):
        self.root = root
        self.prefix = args.topic_prefix
        self.events = queue.Queue()
        self.cards = {}

        root.title("Vision Auth - Device Dashboard")
        root.geometry("960x520")
        root.configure(bg=BG)
        tk.Label(root, text="Real-time Edge Authentication", fg=FG, bg=BG,
                 font=("Segoe UI", 20, "bold")).pack(pady=(14, 4))
        self.status = tk.Label(root, text=f"MQTT: connecting to {args.broker}:{args.port}...", fg=MUTED, bg=BG,
                               font=("Segoe UI", 10))
        self.status.pack()
        self.grid = tk.Frame(root, bg=BG)
        self.grid.pack(fill="both", expand=True, padx=14, pady=14)

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.connect_async(args.broker, args.port, keepalive=30)
        self.client.loop_start()

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.pump)

    # MQTT callbacks run on the network thread: hand everything to Tk via a queue.
    def on_connect(self, client, userdata, flags, reason_code, properties):
        client.subscribe(f"{self.prefix}/+/inference")
        client.subscribe(f"{self.prefix}/+/status")
        self.events.put(("mqtt", f"MQTT: connected ({reason_code})"))

    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.events.put(("mqtt", f"MQTT: disconnected ({reason_code}), retrying..."))

    def on_message(self, client, userdata, msg):
        parts = msg.topic.split("/")
        if len(parts) < 3:
            return
        device, kind = parts[-2], parts[-1]
        payload = msg.payload.decode("utf-8", errors="replace")
        if kind == "status":
            self.events.put(("status", device, payload.strip() == "online"))
        elif kind == "inference":
            try:
                self.events.put(("inference", device, json.loads(payload)))
            except json.JSONDecodeError:
                # Legacy Pi firmware published the bare class name.
                self.events.put(("inference", device, {"label": payload, "confidence": 1.0, "accepted": True}))

    def card(self, device):
        if device not in self.cards:
            card = DeviceCard(self.grid, device)
            n = len(self.cards)
            card.frame.grid(row=n // 2, column=n % 2, sticky="nsew", padx=8, pady=8)
            self.grid.columnconfigure(n % 2, weight=1)
            self.cards[device] = card
        return self.cards[device]

    def pump(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event[0] == "mqtt":
                self.status.config(text=event[1])
            elif event[0] == "status":
                self.card(event[1]).set_status(event[2])
            elif event[0] == "inference":
                self.card(event[1]).update(event[2])
        for card in self.cards.values():
            card.check_stale()
        self.root.after(100, self.pump)

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic-prefix", default="vision-auth")
    args = parser.parse_args()

    root = tk.Tk()
    Dashboard(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
