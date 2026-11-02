"""Post-hoc diagnostics on frozen inference. These are NOT official nuScenes metrics."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

from .detections import TRACKING_CLASSES


def category(name):
    if name.startswith('human.pedestrian.'):
        return 'pedestrian'
    if name.startswith('vehicle.bus.'):
        return 'bus'
    value = name.removeprefix('vehicle.')
    return value if value in TRACKING_CLASSES else None


def match_centers(preds, truth, threshold=2.0):
    """Maximum valid matches, then minimum center distance, class-consistent."""
    if not preds or not truth:
        return []
    distance = np.linalg.norm(np.array([p['translation'][:2] for p in preds])[:,None,:] -
                              np.array([g['translation'][:2] for g in truth])[None,:,:], axis=2)
    same = np.array([[p['tracking_name'] == g['tracking_name'] for g in truth] for p in preds])
    valid = (distance <= threshold) & same
    cost = np.where(valid, distance, 1e6)
    a, b = linear_sum_assignment(cost)
    return [(int(i), int(j), float(distance[i,j])) for i,j in zip(a,b) if valid[i,j]]


def evaluate(tracks, data_root):
    meta = Path(data_root)/'v1.0-mini'
    annotations = json.loads((meta/'sample_annotation.json').read_text())
    instances = {x['token']: x for x in json.loads((meta/'instance.json').read_text())}
    categories = {x['token']: x['name'] for x in json.loads((meta/'category.json').read_text())}
    samples = {x['token']: x for x in json.loads((meta/'sample.json').read_text())}
    sample_data = {x['token']: x for x in json.loads((meta/'sample_data.json').read_text())}
    ego_poses = {x['token']: x for x in json.loads((meta/'ego_pose.json').read_text())}
    ego_by_sample = {sd['sample_token']: ego_poses[sd['ego_pose_token']]['translation']
                     for sd in sample_data.values() if sd['is_key_frame']}
    # Use the same LIDAR_TOP ego pose for range gating, as inference does.
    from .detections import read_mini
    _, lookup = read_mini(data_root)
    for token,s in lookup['sample'].items():
        sd = lookup['sample_data'][s['data']['LIDAR_TOP']]
        ego_by_sample[token] = ego_poses[sd['ego_pose_token']]['translation']
    gt = {}
    for ann in annotations:
        name = category(categories[instances[ann['instance_token']]['category_token']])
        if name is None or ann['num_lidar_pts'] <= 0:
            continue
        ego = ego_by_sample[ann['sample_token']]
        if np.linalg.norm(np.array(ann['translation'][:2])-ego[:2]) > 50:
            continue
        gt.setdefault(ann['sample_token'], []).append({'translation': ann['translation'],
            'tracking_name': name, 'tracking_id': ann['instance_token']})
    report = {'source_sha256': tracks['source_sha256'], 'source': tracks['source'],
              'metric': 'Custom class-consistent one-to-one XY center distance <=2m; NOT AMOTA/recall from official evaluator',
              'scope': 'Seven tracking classes; GT with >0 LiDAR returns; GT and predictions within 50m of LIDAR_TOP ego pose; all class ranges 50m.',
              'identity_metric': 'ID changes for a GT instance between successive matched observations, including gaps; not official IDSW.',
              'prediction_policy': 'Prediction-only states included in lifecycle mode regardless of decayed score; tracks born from >=0.1 detections.',
              'comparison': 'Detector baseline uses the same confidence >=0.1 and upstream class-aware NMS. Historical YOLO-seeded 2D proxy scores are not comparable.',
              'scenes': {}, 'aggregate': {}}
    aggregate = {}
    for name, scene in tracks['scenes'].items():
        start, end = tracks['gap']['start_index'], tracks['gap']['end_index_exclusive']
        seeds = deepcopy(scene['normal'][scene['tokens'][start-1]]['tracks']) if start else []
        variants = {
            'detector_baseline': [(t, [{**d, 'tracking_name': d['detection_name']} for d in scene['normal'][t]['detections']]) for t in scene['tokens']],
            'simpletrack_detected': [(t,[p for p in scene['normal'][t]['tracks'] if p['status']=='detected']) for t in scene['tokens']],
            'simpletrack_lifecycle': [(t,scene['normal'][t]['tracks']) for t in scene['tokens']],
            'gap_simpletrack': [(t,scene['detection_gap'][t]['tracks']) for t in scene['tokens'][start:end]],
            'gap_carry_only': [(t,seeds) for t in scene['tokens'][start:end]],
        }
        scene_report = {}
        for variant, frames in variants.items():
            summary = Counter(frames=0, gt_boxes=0, prediction_boxes=0, matched=0, unmatched_predictions=0, missed_gt=0, id_changes=0, match_resumptions=0)
            last_ids, last_frames = {}, {}
            errors, frame_reports = [], []
            for index,(token, predictions) in enumerate(frames):
                ego = np.array(ego_by_sample[token][:2])
                predictions = [p for p in predictions if np.linalg.norm(np.array(p['translation'][:2])-ego) <= 50]
                truth = gt.get(token, [])
                matches = match_centers(predictions, truth)
                changes, resumes = 0, 0
                for i,j,error in matches:
                    errors.append(error)
                    if 'tracking_id' in predictions[i]:
                        tid, gid = predictions[i]['tracking_id'], truth[j]['tracking_id']
                        changes += gid in last_ids and last_ids[gid] != tid
                        resumes += gid in last_frames and last_frames[gid] < index-1
                        last_ids[gid], last_frames[gid] = tid, index
                counts = {'frames': 1, 'gt_boxes': len(truth), 'prediction_boxes': len(predictions), 'matched': len(matches),
                          'unmatched_predictions': len(predictions)-len(matches), 'missed_gt': len(truth)-len(matches),
                          'id_changes': changes, 'match_resumptions': resumes}
                summary.update(counts)
                frame_reports.append({'sample_token': token, **counts})
            result = dict(summary)
            result.update({'matched_fraction_of_gt': summary['matched']/summary['gt_boxes'] if summary['gt_boxes'] else None,
                           'matched_fraction_of_predictions': summary['matched']/summary['prediction_boxes'] if summary['prediction_boxes'] else None,
                           'mean_matched_xy_error_m': float(np.mean(errors)) if errors else None,
                           'identity_available': variant != 'detector_baseline', 'per_frame': frame_reports})
            scene_report[variant] = result
            agg = aggregate.setdefault(variant, Counter())
            agg.update(summary)
            agg.update({'xy_error_sum': sum(errors)})
        scene_report['lost_events'] = sum(e['event']=='lost' for e in scene['events'])
        scene_report['gap_lost_events'] = sum(e['event']=='lost' for e in scene['gap_events'])
        report['scenes'][name] = scene_report
    for variant, counts in aggregate.items():
        d = dict(counts)
        d['matched_fraction_of_gt'] = d['matched']/d['gt_boxes'] if d['gt_boxes'] else None
        d['matched_fraction_of_predictions'] = d['matched']/d['prediction_boxes'] if d['prediction_boxes'] else None
        d['mean_matched_xy_error_m'] = d.pop('xy_error_sum')/d['matched'] if d['matched'] else None
        report['aggregate'][variant] = d
    return report
