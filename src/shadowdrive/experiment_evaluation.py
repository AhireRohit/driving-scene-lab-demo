"""Annotation access lives exclusively in this post-freeze evaluator."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .evaluation import category,match_centers
from .detections import read_mini,scene_samples

METRIC_KEYS=('frames','gt_boxes','prediction_boxes','matched','missed_gt','unmatched_predictions',
             'identity_changes','match_resumptions','seed_id_retained','seeded_gt_visible',
             'stale_prediction_boxes','exact_duplicate_boxes','duplicate_track_ids',
             'near_center_duplicate_candidates','camera_supported')


def load_truth(root, scene_names):
    root=Path(root);meta=root/'v1.0-mini'
    tables,lookup=read_mini(root)
    wanted={s['token'] for scene in tables['scene'] if scene['name'] in scene_names for s in scene_samples(scene,lookup['sample'])}
    instances={x['token']:x for x in json.loads((meta/'instance.json').read_text())}
    cats={x['token']:x['name'] for x in json.loads((meta/'category.json').read_text())}
    ego={}
    for token in wanted:
        sd=lookup['sample_data'][lookup['sample'][token]['data']['LIDAR_TOP']]
        ego[token]=np.array(lookup['ego_pose'][sd['ego_pose_token']]['translation'][:2])
    truth={t:[] for t in wanted}
    for a in json.loads((meta/'sample_annotation.json').read_text()):
        token=a['sample_token']
        if token not in wanted:continue
        name=category(cats[instances[a['instance_token']]['category_token']])
        if name is None or a['num_lidar_pts']<=0 or np.linalg.norm(np.array(a['translation'][:2])-ego[token])>50:continue
        truth[token].append({'tracking_name':name,'tracking_id':a['instance_token'],'translation':a['translation']})
    return truth,ego


def summarize(frames, truth, ego, start=10, duration=4, method='lidar'):
    total=Counter({k:0 for k in METRIC_KEYS});errors=[];per_frame=[];last_ids={};last_matches={};seed_ids={}
    timestamps={f['frame_index']:f['timestamp_s'] for f in frames.values()}
    for token,frame in frames.items():
        i=frame['frame_index'];raw=frame['carry_tracks'] if method=='carry' else frame['tracks']
        if method=='policy':raw=[p for p in raw if p.get('policy_keep',True)]
        preds=[p for p in raw if np.linalg.norm(np.array(p['translation'][:2])-ego[token])<=50]
        gt=truth[token];matched=match_centers(preds,gt)
        # A common pre-outage seed population comes from unchanged LiDAR outputs.
        if i==start-1:
            seed=[p for p in frame['tracks'] if np.linalg.norm(np.array(p['translation'][:2])-ego[token])<=50]
            seed_ids={gt[j]['tracking_id']:seed[a]['tracking_id'] for a,j,_ in match_centers(seed,gt)}
        ids={};changes=resumptions=retained=0
        for a,j,distance in matched:
            gid=gt[j]['tracking_id'];tid=preds[a]['tracking_id'];ids[gid]=tid;errors.append(distance)
            changes+=gid in last_ids and last_ids[gid]!=tid
            resumptions+=gid in last_matches and last_matches[gid]<i-1
            retained+=seed_ids.get(gid)==tid
            last_ids[gid]=tid;last_matches[gid]=i
        stale=sum(not p.get('lidar_associated',p['status']=='detected') and not p.get('camera_association') and
                  min([age for age in [p.get('age_since_lidar_s'),p.get('age_since_camera_s')] if age is not None] or [0])>1.0 for p in preds)
        geometry=[(p['tracking_name'],*(round(float(x),6) for k in ('translation','size','rotation') for x in p[k])) for p in preds]
        near_pairs=0
        if len(preds)>1:
            centers=np.array([p['translation'][:2] for p in preds])
            distances=np.linalg.norm(centers[:,None,:]-centers[None,:,:],axis=-1)
            same=np.array([[a['tracking_name']==b['tracking_name'] for b in preds] for a in preds])
            near_pairs=int(np.triu((distances<=.25)&same,1).sum())
        counts={'frames':1,'gt_boxes':len(gt),'prediction_boxes':len(preds),'matched':len(matched),
                'missed_gt':len(gt)-len(matched),'unmatched_predictions':len(preds)-len(matched),
                'identity_changes':int(changes),'match_resumptions':int(resumptions),
                'seed_id_retained':int(retained),'seeded_gt_visible':sum(g['tracking_id'] in seed_ids for g in gt),
                'stale_prediction_boxes':stale,'exact_duplicate_boxes':len(geometry)-len(set(geometry)),
                'duplicate_track_ids':len(preds)-len({p['tracking_id'] for p in preds}),
                'near_center_duplicate_candidates':near_pairs,
                'camera_supported':sum(bool(p.get('camera_association')) for p in preds)}
        phase='pre' if i<start else 'outage' if i<start+duration else 'recovery' if i<start+duration+4 else 'post'
        age=i-start+1 if phase=='outage' else i-(start+duration) if phase=='recovery' else None
        age_s=frame['timestamp_s']-timestamps[start-1 if phase=='outage' else start+duration] if phase in ('outage','recovery') else None
        total.update(counts)
        per_frame.append({'sample_token':token,'frame_index':i,'phase':phase,'age_frames':age,'age_s':age_s,**counts})
    result=dict(total);result['f1']=2*total['matched']/(total['gt_boxes']+total['prediction_boxes']) if total['gt_boxes']+total['prediction_boxes'] else 0
    result['mean_xy_error_m']=float(np.mean(errors)) if errors else None
    result['per_frame']=per_frame
    for phase in ('outage','recovery'):
        phase_counts=Counter({k:0 for k in METRIC_KEYS})
        for frame in per_frame:
            if frame['phase']==phase:phase_counts.update({k:frame[k] for k in METRIC_KEYS})
        result[phase]=dict(phase_counts)
    return result


def combine(reports):
    counts=Counter({k:0 for k in METRIC_KEYS})
    for r in reports:
        counts.update({k:v for k,v in r.items() if isinstance(v,int) and k not in ('frame_index','age_frames')})
    output=dict(counts)
    output['f1']=2*counts['matched']/(counts['gt_boxes']+counts['prediction_boxes']) if counts['gt_boxes']+counts['prediction_boxes'] else 0
    return output
