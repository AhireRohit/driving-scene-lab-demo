"""Adapter to actual upstream SimpleTrack MOTModel, not a replacement tracker."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from pyquaternion import Quaternion
import yaml
from .detections import TRACKING_CLASSES


def load_upstream(repo):
    repo = Path(repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from mot_3d.preprocessing import bbox_coarse_hash
    class NumpyCompatibility:
        int = int
        def __getattr__(self, key):
            return getattr(np, key)
    # Scope np.int compatibility to one module; no source/global NumPy mutation.
    bbox_coarse_hash.np = NumpyCompatibility()
    # SciPy's Windows Qhull wrapper creates temporary files per hull. The same
    # 2D convex hull can be computed in memory by GEOS. Return input vertex
    # indices so upstream giou3d and its equations remain unchanged.
    from mot_3d.utils import geometry
    from shapely.geometry import MultiPoint
    class InMemoryHull:
        def __init__(self, points):
            hull = MultiPoint(points).convex_hull
            if hull.geom_type != 'Polygon':
                raise ValueError('Degenerate convex hull')
            self.vertices = np.array([np.where(np.all(points == coordinate, axis=1))[0][0]
                                      for coordinate in list(hull.exterior.coords)[:-1]])
    geometry.ConvexHull = InMemoryHull
    from mot_3d.mot import MOTModel
    from mot_3d.frame_data import FrameData
    from mot_3d.data_protos import BBox, Validity
    from mot_3d.preprocessing import nms
    from mot_3d import association
    from .geometry_backend import giou_distance
    if not hasattr(association, '_shadowdrive_reference_distance'):
        association._shadowdrive_reference_distance = association.compute_iou_distance
    def distance(dets, tracks, asso='iou'):
        if asso == 'giou':
            return giou_distance(dets, tracks)
        return association._shadowdrive_reference_distance(dets, tracks, asso)
    association.compute_iou_distance = distance
    return MOTModel, FrameData, BBox, Validity, nms


def detection_array(box):
    q = Quaternion(box['rotation'])
    w, l, h = box['size']
    return np.array([*box['translation'], q.yaw_pitch_roll[0], l, w, h, box['detection_score']], dtype=float)


def prepare_detections(entries, upstream, threshold=0.1):
    _, _, BBox, _, nms = upstream
    entries = [d for d in entries if d['detection_name'] in TRACKING_CLASSES and d['detection_score'] >= threshold]
    for d in entries:
        if np.prod(d['size']) / (np.prod(d['size']) + 1e-5) <= 0.1:
            raise ValueError('Box volume is too small for upstream NMS numerical stabilization; refusing a non-progress loop')
    boxes = [BBox.array2bbox(detection_array(d)) for d in entries]
    types = [d['detection_name'] for d in entries]
    indices, _ = nms(boxes, types, threshold_low=0.1)
    return [entries[i] for i in indices]


def track_scene(scene, samples, results, repo, config, lookup, gap=None, carry_gap=None, verbose=True):
    upstream = load_upstream(repo)
    MOTModel, FrameData, BBox, Validity, _ = upstream
    models = {name: MOTModel(deepcopy(config)) for name in TRACKING_CLASSES}
    rows, lifecycle = {}, []
    previous = set()
    epoch = 0
    observed_at = {}
    for index, sample in enumerate(samples):
        started = time.perf_counter()
        if verbose and index % 10 == 0:
            print(f'  {scene["name"]} {"detection_gap" if gap else "normal"} frame {index}/{len(samples)}', flush=True)
        token = sample['token']
        stamp = sample['timestamp'] * 1e-6
        if token not in results:
            lifecycle.extend({'frame': index, 'tracking_id': tid, 'event': 'coverage_gap_reset'} for tid in previous)
            models = {name: MOTModel(deepcopy(config)) for name in TRACKING_CLASSES}
            epoch += 1
            previous = set()
            observed_at = {}
            rows[token] = {'frame_index': index, 'timestamp_s': stamp, 'input_status': 'missing_coverage', 'tracks': [], 'detections': []}
            continue
        entries = prepare_detections(results[token], upstream, config['running']['score_threshold'])
        if carry_gap is not None and carry_gap[0] <= index < carry_gap[1]:
            outputs = deepcopy(rows[samples[index-1]['token']]['tracks']) if index else []
            for output in outputs:
                output.update(sample_token=token, status='predicted', upstream_state='carry_memory',
                              valid_official_output=False, lidar_associated=False,
                              age_since_detection_frames=output['age_since_detection_frames']+1,
                              age_since_lidar_s=stamp-output['lidar_observed_at_s'])
            rows[token] = {'frame_index': index, 'timestamp_s': stamp, 'input_status': 'carry_memory_window',
                           'tracks': outputs, 'detections': [], 'original_detection_count': len(entries),
                           'runtime_s': time.perf_counter()-started}
            continue
        gap_active = gap is not None and gap[0] <= index < gap[1]
        used = [] if gap_active else entries
        sd = lookup['sample_data'][sample['data']['LIDAR_TOP']]
        pose = lookup['ego_pose'][sd['ego_pose_token']]
        ego = np.eye(4)
        ego[:3, :3] = Quaternion(pose['rotation']).rotation_matrix
        ego[:3, 3] = pose['translation']
        outputs = []
        for cls in TRACKING_CLASSES:
            dets = [detection_array(d) for d in used if d['detection_name'] == cls]
            data = FrameData(dets=dets, ego=ego, time_stamp=stamp, pc=None,
                             det_types=[cls]*len(dets), aux_info={'is_key_frame': True, 'velos': None})
            tracked = models[cls].frame_mot(data)
            states = {t.id: t for t in models[cls].trackers}
            for box, tid, state, name in tracked:
                trk = states[tid]
                fresh = state.startswith('birth') or (state.startswith('alive_') and state.split('_')[1] == '1')
                ident = f'{scene["name"]}:e{epoch}:{name}:{tid}'
                if fresh:
                    observed_at[ident] = stamp
                velocity = trk.motion_model.kf.x[7:9].reshape(-1).tolist()
                outputs.append({'sample_token': token, 'tracking_id': ident, 'tracking_name': name,
                                'translation': [float(box.x), float(box.y), float(box.z)],
                                'size': [float(box.w), float(box.l), float(box.h)],
                                'rotation': Quaternion(axis=[0,0,1], angle=float(box.o)).elements.tolist(),
                                'velocity': velocity, 'tracking_score': float(box.s),
                                'status': 'detected' if fresh else 'predicted', 'upstream_state': state,
                                'last_detection_score': float(trk.motion_model.score),
                                'valid_official_output': bool(Validity.valid(state)),
                                'age_since_detection_frames': int(trk.life_manager.time_since_update)})
                outputs[-1].update(lidar_associated=fresh, lidar_observed_at_s=observed_at.get(ident),
                                   age_since_lidar_s=stamp-observed_at[ident] if ident in observed_at else None)
        current = {t['tracking_id'] for t in outputs}
        assert len(current) == len(outputs), 'Duplicate track ID within a frame'
        lifecycle.extend({'frame': index, 'tracking_id': tid, 'event': 'lost'} for tid in sorted(previous-current))
        lifecycle.extend({'frame': index, 'tracking_id': tid, 'event': 'born'} for tid in sorted(current-previous))
        previous = current
        rows[token] = {'frame_index': index, 'timestamp_s': stamp,
                       'input_status': 'withheld_detection_experiment' if gap_active else ('present_empty' if not results[token] else 'present'),
                       'tracks': outputs, 'detections': used, 'original_detection_count': len(entries),
                       'runtime_s': time.perf_counter()-started}
    return rows, lifecycle


def run_tracking(payload, coverage, tables, lookup, repo, gap=(10, 14), only_scene=None):
    from .detections import scene_samples
    config = yaml.safe_load((Path(repo) / 'configs/nu_configs/giou.yaml').read_text())
    config['running']['score_threshold'] = 0.1
    config['data_loader']['pc'] = False
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    selected = {x['scene'] for x in coverage['scenes'] if x['complete']}
    if only_scene:
        if only_scene not in selected:
            raise ValueError(f'{only_scene} has incomplete or zero prediction coverage')
        selected = {only_scene}
    if not selected:
        raise ValueError('No fully covered local scene; see coverage.json before tracking')
    output = {'schema_version': 1, 'tracker': 'SimpleTrack MOTModel', 'upstream_commit': commit,
              'source_sha256': coverage['sha256'], 'source': coverage['source'],
              'coordinates': coverage['coordinates'], 'config': config,
              'compatibility': 'Module-local np.int alias; GEOS hull and batched upstream GIoU equation. Equivalence tested against reference SciPy/loop implementations. Upstream checkout unmodified.',
              'notes': ['Class-separated models; actual timestamps in seconds; reset at scene boundaries.',
                        'Source roll/pitch becomes yaw-only, matching upstream nuScenes preprocessing.',
                        'Prediction-only states are exported separately from official valid detection-associated outputs.',
                        'Camera display changes never modify LiDAR inputs. Detector temporal causality is unverified.'],
              'gap': {'start_index': gap[0], 'end_index_exclusive': gap[1], 'meaning': 'Artificial withholding of published detections, not LiDAR corruption'},
              'scenes': {}}
    for scene in sorted(tables['scene'], key=lambda x: x['name']):
        if scene['name'] not in selected:
            continue
        print('SimpleTrack:', scene['name'], flush=True)
        samples = list(scene_samples(scene, lookup['sample']))
        normal, events = track_scene(scene, samples, payload['results'], repo, config, lookup)
        dropout, gap_events = track_scene(scene, samples, payload['results'], repo, config, lookup, gap=gap)
        output['scenes'][scene['name']] = {'scene_token': scene['token'], 'description': scene['description'],
                                         'tokens': [s['token'] for s in samples], 'normal': normal,
                                         'detection_gap': dropout, 'events': events, 'gap_events': gap_events}
    return output
