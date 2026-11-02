"""Redistributable procedural UI demo. No nuScenes files or pretrained predictions."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from .evidence import annotate_frames,projected_box


class DemoSensors:
    def __init__(self,seed):self.seed=seed

    def ego_position(self,token):return np.array([0.,0.,0.])

    def camera_geometry(self,token):
        index=int(token.rsplit(':',1)[1])
        matrix=np.array([[1.,0,0,0],[0,0,-1,1.5],[0,1,0,5],[0,0,0,1]])
        K=np.array([[520.,0,480],[0,520,270],[0,0,1]])
        return matrix,K,(960,540),index*.5

    def camera(self,token):
        matrix,K,_,_=self.camera_geometry(token)
        image=Image.new('RGB',(960,540),'#132538');d=ImageDraw.Draw(image)
        d.polygon([(390,270),(570,270),(940,540),(20,540)],fill='#263942')
        for y in range(300,530,45):d.line((480,y,480,y+20),fill='#aabbbd',width=4)
        d.text((20,20),'PROCEDURAL DEMO / no dataset image / no detector inference',fill='#dfe8ee')
        return image,matrix,K

    def lidar(self,token):
        rng=np.random.default_rng(self.seed)
        xyz=np.c_[rng.uniform(-25,25,650),rng.uniform(-5,55,650),rng.normal(-.2,.03,650)]
        return xyz,np.eye(4)


class DemoSnapshots:
    def __init__(self,bundle):
        self.bundle=bundle;self.manifest={'selection':{'selected':{'confidence':.5,'max_age':5}}}
        self.sensors=DemoSensors(bundle['seed'])

    @lru_cache(maxsize=8)
    def scene(self,name,experiment='clean',duration=4,method='lidar'):
        if name not in self.bundle['scenes']:raise ValueError('Unknown procedural scene')
        if experiment not in ('clean','degraded','blackout','gap','both'):raise ValueError('Unknown condition')
        if int(duration) not in (1,2,4) or method not in ('lidar','carry','policy'):raise ValueError('Unknown demo setting')
        meta=self.bundle['scenes'][name];rows={};packets={};events=[];duration=int(duration)
        tokens=[f'{name}:{i}' for i in range(self.bundle['frames'])]
        for i,token in enumerate(tokens):
            window=10<=i<10+duration
            gap=window and experiment in ('gap','both')
            unavailable=window and experiment in ('blackout','both')
            mode='blackout' if unavailable else 'degraded' if window and experiment=='degraded' else 'clean'
            tracks=[];observations=[]
            position_index=9 if window and method=='carry' else i
            for j in range(5):
                if j==4 and i>=11:
                    if i==11:events.append({'frame':i,'tracking_id':f'{name}:e0:car:4','event':'lost'})
                    continue
                category='pedestrian' if j==3 else 'car'
                p=[-9+j*4+position_index*(.12 if j==3 else .04),12+j*6+meta['offset']+position_index*(.15+.07*j),.9]
                lidar=not gap and not(window and method=='carry') and j!=4
                t={'tracking_id':f'{name}:e0:{category}:{j}','tracking_name':category,'translation':p,
                   'size':[.6,.6,1.7] if category=='pedestrian' else [1.8,4.,1.5],
                   'rotation':[.70710678,0,0,.70710678],'velocity':[0,.4],
                   'tracking_score':.8,'last_detection_score':.8,'status':'detected' if lidar else 'predicted',
                   'upstream_state':'illustrative_current' if lidar else 'illustrative_prediction','lidar_associated':lidar}
                tracks.append(t)
                box=projected_box(t,*self.sensors.camera_geometry(token)[:3])
                if box and not unavailable and j!=4 and not(mode=='degraded' and j%2):
                    observations.append({'id':f'illustration:{j}','xyxy':box,'confidence':.8,'class_name':category,'compatible_classes':[category]})
            rows[token]={'frame_index':i,'timestamp_s':i*.5,'tracks':tracks,'input_status':'withheld_detection_experiment' if gap else 'present'}
            packets[token]={'identity':{'mode':mode,'camera_sample_data_token':token},'cache_key':'procedural','available':not unavailable,'detections':observations}
        annotated=annotate_frames(rows,self.sensors,lambda token,index:packets[token],negative_controls=False)
        for f in annotated.values():
            f['all_tracks']=deepcopy(f['tracks'])
            if method=='policy':f['tracks']=[t for t in f['tracks'] if t['policy_keep']]
            f['policy_suppressed']=len(f['all_tracks'])-len(f['tracks'])
        return {'scene_token':name,'tokens':tokens,'description':meta['description']+' — illustrative geometry, not benchmark results.',
                'normal':annotated,'events':events,'gap_events':events,'cohort':'procedural demo'}

    def frame(self,name,index,experiment='clean',duration=4,method='lidar'):
        scene=self.scene(name,experiment,int(duration),method);index=min(max(int(index or 0),0),len(scene['tokens'])-1)
        token=scene['tokens'][index]
        return scene,index,token,scene['normal'][token],'normal'


def initialize_demo(replay,root):
    bundle=json.loads((Path(root)/'demo/bundle.json').read_text(encoding='utf-8'))
    replay.root=Path(root);replay.demo=True;replay.snapshots=DemoSnapshots(bundle)
    replay.sensors=replay.snapshots.sensors;replay.coverage={'mini_covered':0};replay.benchmark=None;replay.evaluation=None
    replay.tracks={'scenes':{name:replay.snapshots.scene(name) for name in bundle['scenes']}}
