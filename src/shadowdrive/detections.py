"""Strict nuScenes detection ingestion. No annotation table is accessed here."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

TRACKING_CLASSES = ('bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck')
DETECTION_CLASSES = set(TRACKING_CLASSES) | {'barrier', 'traffic_cone', 'construction_vehicle'}
META_KEYS = ('use_camera', 'use_lidar', 'use_radar', 'use_map', 'use_external')
TOKEN = re.compile(r'^[0-9a-f]{32}$')


def numeric_vector(value, length, where):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f'{where}: expected a list of {length} numbers')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f'{where}: values must be finite numbers')
    return value


def validate_frame(token, entries):
    if not isinstance(token, str) or not TOKEN.fullmatch(token):
        raise ValueError(f'Invalid sample token: {token!r}; expected 32 lowercase hex characters')
    if not isinstance(entries, list):
        raise ValueError(f'{token}: detections must be a list (an empty list is valid)')
    if len(entries) > 500:
        raise ValueError(f'{token}: exceeds nuScenes limit of 500 detections per sample')
    for i, box in enumerate(entries):
        where = f'{token}, detection {i}'
        if not isinstance(box, dict):
            raise ValueError(f'{where}: expected an object')
        if box.get('sample_token') != token:
            raise ValueError(f'{where}: sample_token does not match its results key')
        numeric_vector(box.get('translation'), 3, where + ' translation')
        size = numeric_vector(box.get('size'), 3, where + ' size [w,l,h]')
        if min(size) <= 0:
            raise ValueError(f'{where}: all dimensions must be positive')
        rotation = numeric_vector(box.get('rotation'), 4, where + ' rotation [w,x,y,z]')
        norm = math.sqrt(sum(x*x for x in rotation))
        if abs(norm - 1) > 0.001:
            raise ValueError(f'{where}: quaternion must have unit norm; got {norm:.6g}')
        numeric_vector(box.get('velocity'), 2, where + ' velocity [vx,vy] m/s')
        if box.get('detection_name') not in DETECTION_CLASSES:
            raise ValueError(f'{where}: unsupported detection_name {box.get("detection_name")!r}')
        score = box.get('detection_score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'{where}: detection_score must be finite and within [0,1]')


def validate_meta(meta):
    if not isinstance(meta, dict) or any(type(meta.get(k)) is not bool for k in META_KEYS):
        raise ValueError('meta must declare boolean ' + ', '.join(META_KEYS))


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected an object containing meta and results')
    validate_meta(payload.get('meta'))
    results = payload.get('results')
    if not isinstance(results, dict) or not results:
        raise ValueError('results must be a nonempty sample-token mapping')
    names = Counter()
    for token, entries in results.items():
        validate_frame(token, entries)
        names.update(x['detection_name'] for x in entries)
    if not sum(names.values()):
        raise ValueError('Entire artifact contains zero detections; individual empty samples are allowed')
    return len(results), sum(names.values()), names


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_mini(root):
    """Minimal inference metadata: deliberately never load sample_annotation.json."""
    root = Path(root)
    meta = root / 'v1.0-mini'
    tables = {name: json.loads((meta / f'{name}.json').read_text())
              for name in ('scene', 'sample', 'sample_data', 'calibrated_sensor', 'ego_pose')}
    lookups = {name: {row['token']: row for row in rows} for name, rows in tables.items()}
    sensors = {x['token']: x for x in json.loads((meta / 'sensor.json').read_text())}
    for sample in tables['sample']:
        sample['data'] = {}
    for sd in tables['sample_data']:
        if sd['is_key_frame']:
            channel = sensors[lookups['calibrated_sensor'][sd['calibrated_sensor_token']]['sensor_token']]['channel']
            lookups['sample'][sd['sample_token']]['data'][channel] = sd['token']
    return tables, lookups


def scene_samples(scene, lookup):
    sample = lookup[scene['first_sample_token']]
    previous = None
    while True:
        if previous is not None and sample['timestamp'] <= previous:
            raise ValueError(f'{scene["name"]}: non-increasing timestamps')
        yield sample
        previous = sample['timestamp']
        if not sample['next']:
            break
        sample = lookup[sample['next']]


def ingest(path, data_root, provenance_path):
    """Validate all source rows, retaining only local mini tokens in memory."""
    import ijson
    path = Path(path)
    tables, lookup = read_mini(data_root)
    wanted = set(lookup['sample'])
    with path.open('rb') as f:
        meta = next(ijson.items(f, 'meta', use_float=True), None)
    validate_meta(meta)
    retained, seen, classes = {}, set(), Counter()
    total = 0
    with path.open('rb') as f:
        for token, entries in ijson.kvitems(f, 'results', use_float=True):
            if token in seen:
                raise ValueError(f'Duplicate sample key: {token}')
            seen.add(token)
            validate_frame(token, entries)
            classes.update(x['detection_name'] for x in entries)
            total += len(entries)
            if token in wanted:
                retained[token] = entries
    if not seen or not total:
        raise ValueError('Artifact contains no usable detections')
    provenance = json.loads(Path(provenance_path).read_text(encoding='utf8'))
    digest = sha256(path)
    if provenance.get('sha256') != digest:
        raise ValueError('Artifact SHA256 differs from provenance; verify source before continuing')
    coverage = []
    for scene in sorted(tables['scene'], key=lambda x: x['name']):
        samples = list(scene_samples(scene, lookup['sample']))
        covered = sum(s['token'] in retained for s in samples)
        coverage.append({'scene': scene['name'], 'covered': covered, 'total': len(samples),
                         'complete': covered == len(samples),
                         'boxes': sum(len(retained.get(s['token'], [])) for s in samples)})
    report = {'source': provenance, 'sha256': digest, 'source_tokens': len(seen), 'source_boxes': total,
              'source_classes': dict(classes), 'meta': meta, 'scenes': coverage,
              'mini_covered': len(retained), 'mini_total': len(wanted),
              'coordinates': 'global meters; size [width,length,height]; quaternion [w,x,y,z]; velocity global m/s',
              'causality': 'Tracker is forward-only; detector sweep/temporal preprocessing is unspecified by artifact publisher.'}
    return {'meta': meta, 'results': retained}, report, tables, lookup
