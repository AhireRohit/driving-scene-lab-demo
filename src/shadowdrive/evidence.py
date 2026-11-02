"""Causal per-frame cross-sensor association/annotation and output confirmation."""
from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment
from .camera import CLASS_MAP
from .sensors import camera_segments

COLORS={'LIDAR_AND_CAMERA':'#4de2b0','LIDAR_ONLY':'#7cc1fa','CAMERA_ONLY':'#da9cff','PREDICTION_ONLY':'#ffbc66'}
GATES={'min_iou':.10,'max_normalized_center_distance':.75,'min_projected_area_px':64,
       'min_center_depth_m':1.0,'max_sensor_offset_s':.15}


def projected_box(track, matrix, K, image_size, dt=0):
    box=deepcopy(track)
    velocity=np.array([*track.get('velocity',[0,0])[:2],0])
    box['translation']=(np.array(box['translation'])+velocity*dt).tolist()
    center=matrix[:3,:3]@np.array(box['translation'])+matrix[:3,3]
    if center[2]<=GATES['min_center_depth_m']:return None
    segments=camera_segments(box,matrix,K)
    if not segments:return None
    points=np.vstack(segments)
    width,height=image_size
    lo=np.maximum(points.min(axis=0),[0,0]);hi=np.minimum(points.max(axis=0),[width,height])
    if np.any(hi<=lo) or np.prod(hi-lo)<GATES['min_projected_area_px']:return None
    return np.r_[lo,hi].tolist()


def box_quality(a,b):
    a=np.array(a);b=np.array(b)
    wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    intersection=float(np.prod(wh));union=float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)
    iou=intersection/union if union>0 else 0.
    distance=float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/max(np.linalg.norm(a[2:]-a[:2]),1))
    return iou,distance


def associate_camera(tracks, detections, matrix, K, image_size, dt=0, available=True):
    projected=[projected_box(t,matrix,K,image_size,dt) for t in tracks]
    records=[{'camera_visible':b is not None,'camera_class_mapped':t['tracking_name'] in {c for v in CLASS_MAP.values() for c in v},
              'projected_xyxy':b,'camera_association':None} for t,b in zip(tracks,projected)]
    if not available or abs(dt)>GATES['max_sensor_offset_s'] or not tracks or not detections:return records
    cost=np.full((len(tracks),len(detections)),1e6)
    quality={}
    for i,(track,box) in enumerate(zip(tracks,projected)):
        if box is None:continue
        for j,det in enumerate(detections):
            if track['tracking_name'] not in det['compatible_classes']:continue
            iou,distance=box_quality(box,det['xyxy'])
            if iou>=GATES['min_iou'] and distance<=GATES['max_normalized_center_distance']:
                cost[i,j]=1-iou;quality[i,j]=(iou,distance)
    a,b=linear_sum_assignment(cost)
    for i,j in zip(a,b):
        if cost[i,j]>=1e6:continue
        iou,distance=quality[i,j]
        records[i]['camera_association']={'detection_id':detections[j]['id'],'iou':iou,
            'normalized_center_distance':distance,'observation_confidence':detections[j]['confidence']}
    return records


def evidence_state(lidar,camera):
    return 'LIDAR_AND_CAMERA' if lidar and camera else 'LIDAR_ONLY' if lidar else 'CAMERA_ONLY' if camera else 'PREDICTION_ONLY'


def confirmation_policy(track):
    """Output filter only; cannot move boxes, create IDs or update/extend Kalman tracks."""
    eligible=(track['camera_available'] and track['camera_visible'] and track['camera_class_mapped'] and track['camera_timestamp_eligible'])
    if not eligible:return True,'fallback_camera_ineligible'
    if track['camera_association'] is not None:return True,'confirmed_by_current_camera'
    return False,'suppressed_without_current_camera_match'


def annotate_frames(rows, store, packet_for_frame, negative_controls=True):
    output={};last_camera={};last_lidar={}
    for token,frame in rows.items():
        packet=packet_for_frame(token,frame['frame_index'])
        matrix,K,size,camtime=store.camera_geometry(token)
        dt=camtime-frame['timestamp_s'];clock=max(camtime,frame['timestamp_s'])
        available=packet['available']
        if not available and packet['detections']:raise ValueError('Unavailable camera must have zero observations')
        if packet['identity']['mode']=='blackout' and (available or packet['detections']):raise ValueError('Blackout isolation violated')
        tracks=deepcopy(frame['tracks'])
        records=associate_camera(tracks,packet['detections'],matrix,K,size,dt,available)
        for track,record in zip(tracks,records):
            tid=track['tracking_id'];lidar=track.get('lidar_associated',track['status']=='detected')
            camera=record['camera_association'] is not None
            if lidar:last_lidar[tid]=frame['timestamp_s']
            if camera:last_camera[tid]=camtime
            track.update(record)
            track.update(lidar_associated=lidar,camera_available=available,camera_timestamp_eligible=abs(dt)<=GATES['max_sensor_offset_s'],
                         evidence_state=evidence_state(lidar,camera),camera_lidar_offset_s=dt,
                         age_since_lidar_s=clock-last_lidar[tid] if tid in last_lidar else None,
                         age_since_camera_s=clock-last_camera[tid] if tid in last_camera else None,
                         lidar_observed_at_s=last_lidar.get(tid),camera_observed_at_s=last_camera.get(tid))
            track['policy_keep'],track['policy_reason']=confirmation_policy(track)
            track['display_color']=COLORS[track['evidence_state']]
        result={**frame,'tracks':tracks,'evidence_timestamp_s':clock,
                'camera':{'sample_token':token,'available':available,'mode':packet['identity']['mode'],'timestamp_s':camtime,
                          'sample_data_token':packet['identity']['camera_sample_data_token'],
                          'cache_key':packet['cache_key'],'detections':packet['detections'],
                          'unmapped_observations':sum(not d['compatible_classes'] for d in packet['detections'])}}
        if negative_controls:
            result['negative_controls']={}
            for shift in (3,5):
                shifted=deepcopy(frame['tracks'])
                for t in shifted:t['translation'][0]+=shift
                support=associate_camera(shifted,packet['detections'],matrix,K,size,dt,available)
                result['negative_controls'][str(shift)]=[
                    {**{k:t[k] for k in ('translation','size','rotation','tracking_name','tracking_id')},
                     'camera_supported':r['camera_association'] is not None,'camera_visible':r['camera_visible']}
                    for t,r in zip(shifted,support)]
        output[token]=result
    return output
