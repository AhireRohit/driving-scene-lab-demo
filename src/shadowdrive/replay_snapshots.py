"""One synchronized frame source for every replay panel, with bounded lazy loading."""
from functools import lru_cache
from copy import deepcopy
from pathlib import Path
from .storage import read_json
from .detections import sha256

CONDITION_MAP={'clean':'normal','degraded':'degraded','blackout':'blackout','gap':'lidar_gap','both':'both'}


class ScenarioStore:
    def __init__(self,output):
        self.output=Path(output)
        self.manifest=read_json(self.output/'evidence_manifest.json')

    @lru_cache(maxsize=2)
    def load(self,scene,condition,duration):
        item=self.manifest['files'][f'{scene}/{condition}/{duration}']
        path=self.output/item['path']
        if sha256(path)!=item['sha256']:raise ValueError('Replay scenario hash mismatch')
        data=read_json(path)
        if data['scene']!=scene or data['condition']!=condition or data['duration_frames']!=duration:
            raise ValueError('Replay scenario metadata mismatch')
        if 'selection' in self.manifest and data.get('selection_fingerprint')!=self.manifest['selection']['selection_fingerprint']:
            raise ValueError('Replay selection fingerprint mismatch')
        return data

    def scene(self,name,experiment='clean',duration=4,method='lidar'):
        if method not in ('lidar','carry','policy'):raise ValueError('Invalid replay method')
        duration=int(duration);condition=CONDITION_MAP[experiment]
        data=self.load(name,condition,duration)
        rows={}
        for token,frame in data['frames'].items():
            if frame['camera']['sample_token']!=token:raise ValueError('Replay camera/sample mismatch')
            tracks=frame['carry_tracks'] if method=='carry' else frame['tracks']
            visible=[t for t in tracks if t['policy_keep']] if method=='policy' else tracks
            rows[token]={**frame,'tracks':visible,'all_tracks':tracks,'policy_suppressed':len(tracks)-len(visible)}
        meta=self.manifest['scenes'][name]
        events=data['carry_events'] if method=='carry' else data['events']
        return {**meta,'normal':rows,'events':events,'gap_events':events,
                'condition':condition,'duration_frames':duration,'method':method}

    def frame(self,name,index,experiment='clean',duration=4,method='lidar'):
        scene=self.scene(name,experiment,duration,method)
        index=min(max(int(index or 0),0),len(scene['tokens'])-1)
        token=scene['tokens'][index]
        return scene,index,token,scene['normal'][token],'normal'
