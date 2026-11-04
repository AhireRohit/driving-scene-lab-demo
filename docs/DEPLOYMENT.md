# Original recorded replay deployment

Live application: https://driving-scene-lab.onrender.com/ . Deployment source/assets: https://github.com/AhireRohit/driving-scene-lab-demo . Full research history: https://github.com/AhireRohit/driving-scene-lab . The legacy repository suffix does not indicate procedural content: the hosted entrypoint now requires the recorded bundle.

## What is packaged

All 162 frames of scenes 0103, 0553, 0796 and 0916 are retained. `recorded_bundle/` contains real clean CAM_FRONT photographs and deterministic degraded variants, recorded LiDAR display points, complete camera/LiDAR transforms and timestamps, 60 byte-preserved inference/evidence scenario caches, 486 original YOLO camera packets, the original CenterPoint-labelled subset and SimpleTrack baseline records, selection fingerprints, evaluation and provenance manifests.

Camera photographs are width 960, JPEG quality 85. Cloud points are sampled with the original viewer's stride, up to 6,500 per scan, and stored as float32 global coordinates. All points are sampled from the actual recordings; none are generated. No frames, scenes or track states are removed. Camera intrinsics and observation rectangles are scaled for display only. Tracker geometry, IDs, association decisions, evidence ages, outage/recovery windows and metrics are unchanged. The full source dataset remains local.

Every asset is hashed. `manifest.json` stores relative paths, original sensor SHA256 values, source/display point counts and image dimensions. The bundle is about 95 MB; no Windows drive is required at runtime. Two scenario files and bounded point caches are held lazily, rather than loading the 35 MB original baseline into Python objects at startup. The baseline is retained compressed for audit.

## Rebuild and test

```powershell
.\.venv-tracking\Scripts\python.exe scripts\build_recorded_bundle.py
.\.venv-tracking\Scripts\python.exe scripts\launch_replay.py --bundle outputs/recorded_bundle --port 8052
.\.venv-tracking\Scripts\python.exe -m pytest -q
.\.venv-tracking\Scripts\python.exe scripts\prepare_dash_deployment.py
```

The preparation script copies source and the validated bundle to the existing deployment checkout under `outputs/dash-deployment`. It removes the superseded generated fixture from that checkout. It never copies datasets, environments, model weights, credentials or the full raw detection artifact. The covered prediction subset, original baseline and actual observation packets are included deliberately as attributed research records.

## Render

The existing service remains `srv-dbk99bo473hc7380vhk0`, free plan, Frankfurt, one instance, one Gunicorn worker with four threads. Build: `pip install -r requirements-demo.txt` (legacy filename retained). Start: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:server`. Automatic deployments are off; a manual deployment is required after pushing runtime/assets.

`app.py` requires `recorded_bundle` beside it, or an explicit `RECORDED_BUNDLE` directory. `/healthz` reports `recorded-replay`, four scene names and 162 frames. Missing, malformed or checksum-invalid assets serve a clear 503 configuration error. No generated scene is substituted. The Dockerfile also copies the real bundle and uses the same entrypoint. No paid plan, GPU, database or persistent disk is required. Free instances can sleep; the original recordings return after cold start.

The old static procedural Space is retired to a redirect to this same application, so there is one replay experience. Historical procedural source remains in Git history, not in the active deployment.

## Terms and verification

See [recorded data credits and controlling terms](RECORDED_DATA_LICENSE.md). The exact first-party terms module was retrieved and read. The non-commercial derivative bundle retains CC BY-NC-SA 4.0 plus the nuScenes Dataset Terms, attribution and change descriptions. No general public-hosting prohibition was found. Original detector checkpoint, sweeps and training overlap remain unknown; those facts do not change the replay into generated content.

Verification results and live screenshots are recorded in `docs/release/recorded_deployment_verification.json` after rollout. Tests cover original cache equivalence, all four scenes, scaled photographs/overlays, real cloud data, blackout isolation, source/path safety and explicit missing-asset errors.

## Verified rollout — 3 November 2026

The original application is live at https://driving-scene-lab.onrender.com/ . Public deployment source and portable assets: https://github.com/AhireRohit/driving-scene-lab-demo . Full research history remains at https://github.com/AhireRohit/driving-scene-lab (private). The legacy deployment repository suffix is retained; it now serves recorded scenes only. The old Hugging Face static application redirects to the same Render replay.

All four scenes (0103, 0553, 0796, 0916), 162 frames, real CAM_FRONT photographs, degraded versions, sampled recorded clouds, calibration/timestamps, 486 YOLO packets and 60 byte-preserved scenario caches are packaged. Saved CenterPoint/SimpleTrack results and evidence are unchanged. The 95,065,819-byte bundle uses 960-pixel photographs and at most 6,500 recorded points per scan for display. No scene/frame was removed. Clean reference images remain display-only during blackout. Missing or corrupt assets return an explicit configuration error.

Validation: 39 local tests passed; all 46 live HTTP cases passed, including photo comparisons against local assets, every scene/condition, alternate methods/durations, blackout isolation, recovery and transport callbacks. Live browser checks verified the four options, playback/pause, timeline scrubbing, BEV, selected-track ages/association, combined outage and recovery. Scene-0916 frame 10 had zero camera/LiDAR observations; frame 14 recovered 49 LiDAR and 5 camera associations. Screenshots and structured reports are in docs/media/live-recorded-*.jpg and docs/release/recorded_*verification.json.

Render runtime commit: 86f873f9650993c5307be48d04bd2a39ec48a242; successful deploy: dep-dbkfn1jm8hqs73bb2nl0. Git asset normalization preserves all 1,043 bundled files byte-for-byte. The first rollout exposed a CRLF checksum error, corrected before verification. A stale procedural browser tab also sent removed scene names during rollout; a reload selects the recorded application. Render retains its existing TCP readiness probe; /healthz is separately verified, and render.yaml specifies HTTP health checks for a future Blueprint application.

The existing service remains on the free plan. Sampled Render memory peaked at 302,702,600 bytes against a 536,870,900-byte limit during the 2 November 21:08–21:30 UTC verification window. This is a sampled observation, not a multi-user load guarantee. Free-tier cold starts and slower playback than local execution remain possible. No paid resource was provisioned.

Publication terms were checked directly: non-commercial data derivatives retain CC BY-NC-SA 4.0 plus the additional nuScenes Dataset Terms, credit and display-change notices. See docs/RECORDED_DATA_LICENSE.md and docs/release/recorded_publication_audit.json. Exact original detector checkpoint/configuration/sweeps/training overlap remain unknown; MCTrack supplies no separate prediction-artifact license, and no license for its software is inferred. No detector weights or full dataset were published. The experimental camera policy is not an established accuracy improvement. Earlier procedural deployment notes below are historical and superseded.

## Playback update fix — 4 November 2026

The current runtime is `173ed48ad60b980324412040618c83344f197abc`, deploy `dep-dbl60iui0phs738jl450`. See [scene/playback fix](PLAYBACK_UPDATE_FIX.md) and [live recheck](release/playback_update_verification.json). Transport is clientside and completion-gated. Existing browser tabs must reload once after the callback graph changes. The free-tier cold start is separate from scene updates in an awake instance. Original recorded assets and all tracking/evidence records are unchanged.
