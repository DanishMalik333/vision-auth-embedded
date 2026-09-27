# MQTT protocol

All devices share one topic layout and payload schema, so the dashboard treats a Raspberry Pi and an ESP32 the same way.

| Topic | Payload | QoS / retain |
|---|---|---|
| `vision-auth/<device-id>/inference` | JSON, one message per inference | 0 / no |
| `vision-auth/<device-id>/status` | `online` or `offline` (`offline` is also the MQTT last will) | 1 / yes |

The `vision-auth` prefix can be changed with `--topic-prefix` (Pi, dashboard) or in `menuconfig` (ESP32).

## Inference payload

```json
{
  "device": "esp32-s3-eye",
  "platform": "esp32s3",
  "seq": 1024,
  "label": "CEO",
  "class_id": 2,
  "confidence": 0.9431,
  "accepted": true,
  "latency_ms": { "pre": 4.1, "infer": 118.7, "post": 0.05, "total": 131.2 },
  "fps": 1.58,
  "uptime_s": 642,
  "diag": { "free_heap": 183204, "min_free_heap": 151880, "free_psram": 6520112, "rssi": -57 }
}
```

| Field | Meaning |
|---|---|
| `label` | Predicted class, or `Unknown` when `confidence` is below the unknown threshold (default 0.70) |
| `accepted` | `true` when `confidence` is at or above the accept threshold (default 0.90), which grants access |
| `latency_ms` | Preprocessing, model inference, postprocessing and end-to-end time including camera capture |
| `fps` | Inference rate, including the configured delay between frames |
| `diag` | Device health. ESP32: internal heap, lowest heap seen, free PSRAM and Wi-Fi RSSI. Pi: `cpu_temp_c`, `cpu_percent` and `mem_percent` |

## Quick checks

```bash
mosquitto_sub -h <broker> -t 'vision-auth/#' -v                           # watch all traffic
mosquitto_pub -h <broker> -t vision-auth/test/status -m online -r         # add a fake device card
```
