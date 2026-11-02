# Driving Scene Lab — 3D Tracking & Sensor Failure Analysis

Source-only deployment mirror for the procedural **Python Dash** demo. The full research repository remains at https://github.com/AhireRohit/driving-scene-lab (private pending historical-media redistribution review). `SOURCE.json` pins its source commit; the full research history is preserved there.

This repository contains no nuScenes data, imagery, pretrained predictions, model weights, benchmark outputs, credentials or working environments. The geometric fixture in `demo/bundle.json` is CC0 and explicitly illustrative. It exercises the same replay/evidence interface without claiming detector accuracy or upstream tracking performance. Internal Python package names remain unchanged.

Run `pip install -r requirements-demo.txt`, then `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 app:server`. `render.yaml` selects the free plan. `/healthz` returns the demo mode. A Dockerfile is also included. The live Python Dash demo is https://driving-scene-lab.onrender.com/. The free static alternative is https://roh24-driving-scene-lab.static.hf.space/.

Upstream attribution: CenterPoint, SimpleTrack, Ultralytics YOLO11s and nuScenes belong to their respective authors; none of their datasets/checkpoints is bundled here. The procedural demo does not run their detectors or the SimpleTrack tracker.
