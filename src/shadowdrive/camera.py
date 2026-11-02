"""Content-addressed actual camera inference cache, independent of tracking/GT."""
from pathlib import Path
import time
import numpy as np
from .detections import sha256
from .sensors import corrupt_camera
from .storage import fingerprint, read_json, write_json

CLASS_MAP={'person':['pedestrian'],'bicycle':['bicycle'],'car':['car'],'motorcycle':['motorcycle'],
           'bus':['bus'],'truck':['truck']}
CORRUPTION={'version':1,'blur_radius':3.0,'brightness':.55,'noise_std':12.,'occlusion_fraction':.12,'seed':42}
SETTINGS={'imgsz':640,'conf':.20,'iou':.50,'max_det':300,'half':False,'augment':False}


def cache_identity(sample_token, sensor_token, timestamp_us, model_sha256, image_sha256, mode, versions, settings=None):
    if mode not in ('clean','degraded','blackout'):raise ValueError('Unknown camera condition')
    return {'schema':1,'sample_token':sample_token,'camera_sample_data_token':sensor_token,
            'timestamp_us':timestamp_us,'model_sha256':model_sha256,'image_sha256':image_sha256,
            'mode':mode,'settings':settings or SETTINGS,'corruption':CORRUPTION if mode=='degraded' else None,
            'versions':versions,'class_map':CLASS_MAP}


def cached_observation(identity, cache_dir, image_factory, predictor):
    key=fingerprint(identity);path=Path(cache_dir)/(key+'.json')
    if path.exists():
        packet=read_json(path)
        if packet.get('identity')!=identity or packet.get('cache_key')!=key:
            raise ValueError(f'Camera cache identity mismatch: {path}')
        if identity['mode']=='blackout' and (packet.get('available') or packet.get('detections')):
            raise ValueError(f'Blackout cache contains camera observations: {path}')
        return packet,True
    available=identity['mode']!='blackout'
    started=time.perf_counter()
    # Blackout never opens pixels and never invokes the predictor.
    detections=predictor(corrupt_camera(image_factory(),identity['mode'])) if available else []
    packet={'cache_key':key,'identity':identity,'available':available,'detections':detections,
            'runtime_s':time.perf_counter()-started,'inference_performed':available}
    write_json(path,packet)
    return packet,False


def prepare_camera_cache(root, tokens, store, model_path, cache_dir, device='0'):
    # Only the existing camera environment needs torch/ultralytics.
    import torch
    import ultralytics
    from ultralytics import YOLO
    model_path=Path(model_path);model_hash=sha256(model_path)
    model=YOLO(str(model_path))
    versions={'torch':torch.__version__,'ultralytics':ultralytics.__version__,'device':str(device)}
    runtime=[];hits=0;manifest={}
    def predict(image):
        # FP32 is the default; avoid the deprecated explicit half=False keyword.
        output=model.predict(image,device=device,verbose=False,**{k:v for k,v in SETTINGS.items() if k!='half'})[0]
        rows=[]
        for i,b in enumerate(output.boxes):
            cls=int(b.cls.item());name=model.names[cls]
            rows.append({'id':i,'class_id':cls,'class_name':name,'compatible_classes':CLASS_MAP.get(name,[]),
                         'xyxy':b.xyxy[0].cpu().tolist(),'confidence':float(b.conf.item())})
        return rows
    for i,token in enumerate(tokens):
        sample=store.lookup['sample'][token]
        sd=store.lookup['sample_data'][sample['data']['CAM_FRONT']]
        image_hash=sha256(Path(root)/sd['filename'])
        for mode in ('clean','degraded','blackout'):
            identity=cache_identity(token,sd['token'],sd['timestamp'],model_hash,image_hash,mode,versions)
            packet,hit=cached_observation(identity,cache_dir,lambda:store.camera(token)[0],predict)
            manifest.setdefault(token,{})[mode]=packet['cache_key']+'.json'
            hits+=hit
            if not hit and packet['inference_performed']:runtime.append(packet['runtime_s'])
        if i%20==0:print(f'YOLO cache {i+1}/{len(tokens)}',flush=True)
    return {'model':str(model_path.name),'model_sha256':model_hash,'settings':SETTINGS,'versions':versions,
            'corruption':CORRUPTION,'frames':manifest,'cache_hits':hits,
            'new_inference_frames':len(runtime),'new_inference_runtime_s':sum(runtime),
            'peak_cuda_allocated_bytes':int(torch.cuda.max_memory_allocated()) if str(device)!='cpu' else None}
