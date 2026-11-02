# Recorded replay: data credits and terms

This is a non-commercial research demonstration of recorded nuScenes mini scenes 0103, 0553, 0796 and 0916 (162 frames). It is not endorsed by, sponsored by, or affiliated with the dataset or algorithm authors.

## nuScenes recordings and derivatives

Camera photographs, LiDAR samples, calibration and timestamps originate from **nuScenes / Motional AD Inc. and affiliates**. Credit: Holger Caesar et al., *nuScenes: A multimodal dataset for autonomous driving*, CVPR 2020. Dataset: https://www.nuscenes.org/nuscenes

The official non-commercial Dataset Terms state that, unless specifically labelled otherwise, the datasets are provided under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**, with additional Dataset Terms that prevail in a conflict. Those terms also apply to derived data. This permits non-commercial sharing and adaptation subject to attribution, license/terms retention, indication of changes and share-alike. It is not a blanket prohibition on public non-commercial hosting.

- Controlling Dataset Terms: https://www.nuscenes.org/terms-of-use
- CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
- Legal code: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Direct check on 2 November 2026: the terms page renders its legal content from https://www.nuscenes.org/static/js/TermsOfUseNonCommercial.d9d49b0c.chunk.js . That exact first-party page module was retrieved and read; it identifies the terms as last updated 16 November 2021. Relevant additional conditions include third-party exceptions where specified in metadata, no implied endorsement/association and compliance with privacy and other applicable laws. No separate restriction was found in the packaged mini sensor metadata that prohibits this non-commercial replay.

**Changes to recordings for hosting:** CAM_FRONT photographs and their deterministic degraded variants are resized to width 960 and JPEG-encoded at quality 85. Point clouds contain actual recorded points sampled with the original viewer's stride, at most 6,500 per scan. No scene, frame, object, box or road was procedurally generated. Calibration intrinsics and observation overlays are scaled only for display. Original source hashes, timestamps, transforms and reduction parameters are retained in the asset manifest. All 162 frames and all 60 original scenario caches are retained. Data derivatives in this replay are distributed under CC BY-NC-SA 4.0 plus the Dataset Terms, without any suggestion that the original recordings were created by this project.

## Saved model results

CenterPoint-labelled detections were published by **Megvii Research / MCTrack**: https://github.com/megvii-research/MCTrack . The publisher's README explicitly provides the KITTI/nuScenes converted data download used here; it separately excludes Waymo converted data for copyright reasons. This package uses only the covered nuScenes subset, not Waymo data. The original source JSON SHA256 is `4186f748f91e071c1276287244e23a01da5908a373af85ee2dfb08ea935b903e`.

Exact CenterPoint checkpoint, detector configuration, input sweeps and training overlap remain unspecified. The MCTrack repository root has no separate LICENSE file, and its download does not provide a distinct prediction-artifact license. No license for MCTrack software is inferred or granted here, and no MCTrack implementation or detector checkpoint is bundled. The stored numeric detections, project-generated tracks and evidence are attributed nuScenes-derived research records, subject to the dataset terms; this notice does not remove any independently applicable third-party rights.

Tracking results were computed using **SimpleTrack** (Pang et al.; upstream MIT notice retained), with **CenterPoint** (Yin et al.) as the attributed detector family. Camera observations were generated locally with **Ultralytics YOLO11s**. The hosted application serves saved results only; it does not redistribute Ultralytics software/model weights or run a detector. Sources:

- https://github.com/tusen-ai/SimpleTrack
- https://github.com/tianweiy/CenterPoint
- https://github.com/ultralytics/ultralytics

No model weights, full nuScenes dataset, GT annotation table or training data are included. This notice licenses the recorded data derivatives as described above; it does not relicense third-party software or imply verified detector provenance.
