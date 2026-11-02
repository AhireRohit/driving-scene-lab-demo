"""Portable assets for the original recorded replay; never generates replacement scenes."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from .storage import read_json
from .replay_snapshots import ScenarioStore
from .detections import sha256

SCENES={'scene-0103','scene-0553','scene-0796','scene-0916'}


def asset_path(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Recorded asset path escapes its bundle')
    if not path.is_file():raise FileNotFoundError(f'Recorded replay asset missing: {relative}')
    return path


class RecordedSensors:
    def __init__(self,root,manifest):
        self.root=Path(root);self.frames=manifest['frames']

    @lru_cache(maxsize=24)
    def lidar(self,token):
        f=self.frames[token]
        with np.load(asset_path(self.root,f['lidar']),allow_pickle=False) as data:xyz=data['xyz']
        return xyz,np.array(f['global_from_lidar'])

    def camera(self,token):
        f=self.frames[token]
        with Image.open(asset_path(self.root,f['clean'])) as image:result=image.convert('RGB')
        K=np.array(f['camera_intrinsic'],dtype=float);K[0,:]*=f['display_scale'][0];K[1,:]*=f['display_scale'][1]
        return result,np.array(f['camera_from_world']),K

    def conditioned_camera(self,token,mode):
        f=self.frames[token]
        if mode=='blackout':return Image.new('RGB',tuple(f['display_size']),'#080d14')
        if mode not in ('clean','degraded'):raise ValueError('Unknown recorded camera condition')
        with Image.open(asset_path(self.root,f[mode])) as image:return image.convert('RGB')

    def display_observations(self,token,detections):
        sx,sy=self.frames[token]['display_scale'];result=deepcopy(detections)
        for d in result:d['xyxy']=[v*s for v,s in zip(d['xyxy'],(sx,sy,sx,sy))]
        return result

    def ego_position(self,token):return np.array(self.frames[token]['ego_position'])


def initialize_recorded(replay,root,bundle):
    bundle=Path(bundle)
    manifest=read_json(asset_path(bundle,'manifest.json'))
    if manifest.get('kind')!='recorded-nuscenes-replay' or set(manifest['scenes'])!=SCENES:
        raise ValueError('Recorded replay requires the four covered nuScenes scenes; procedural data is not accepted')
    if len(manifest['frames'])!=162:raise ValueError('Recorded bundle must contain all 162 covered frames')
    for relative,entry in manifest['files'].items():
        if sha256(asset_path(bundle,relative))!=entry['sha256']:raise ValueError(f'Recorded asset checksum mismatch: {relative}')
    for frame in manifest['frames'].values():
        for field in ('clean','degraded','lidar'):asset_path(bundle,frame[field])
    replay.root=Path(root);replay.output=bundle/'evidence';replay.demo=False
    replay.snapshots=ScenarioStore(replay.output)
    if set(replay.snapshots.manifest['scenes'])!=SCENES:raise ValueError('Recorded scenario coverage mismatch')
    for item in replay.snapshots.manifest['files'].values():asset_path(replay.output,item['path'])
    replay.tracks={'scenes':replay.snapshots.manifest['scenes']}
    replay.coverage=read_json(asset_path(bundle,'baseline/coverage.json'))
    replay.benchmark=read_json(asset_path(bundle,'evidence/benchmark_evaluation.json'))
    if replay.benchmark['selection_fingerprint']!=replay.snapshots.manifest['selection']['selection_fingerprint']:
        raise ValueError('Recorded evaluation/selection mismatch')
    replay.evaluation=None;replay.sensors=RecordedSensors(bundle,manifest)
    replay.recorded_manifest=manifest
