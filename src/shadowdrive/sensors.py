"""Sensor geometry and deterministic display corruption; no GT annotation access."""
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from pyquaternion import Quaternion

EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def transform(translation, rotation):
    out = np.eye(4)
    out[:3,:3] = Quaternion(rotation).rotation_matrix
    out[:3,3] = translation
    return out


def global_from_sensor(sd, lookup):
    pose = lookup['ego_pose'][sd['ego_pose_token']]
    cal = lookup['calibrated_sensor'][sd['calibrated_sensor_token']]
    return transform(pose['translation'], pose['rotation']) @ transform(cal['translation'], cal['rotation'])


def corners(box):
    w,l,h = box['size']
    xyz = np.array([[l/2,w/2,h/2],[l/2,-w/2,h/2],[-l/2,-w/2,h/2],[-l/2,w/2,h/2],
                    [l/2,w/2,-h/2],[l/2,-w/2,-h/2],[-l/2,-w/2,-h/2],[-l/2,w/2,-h/2]])
    return xyz @ Quaternion(box['rotation']).rotation_matrix.T + box['translation']


def camera_segments(box, camera_from_world, intrinsics):
    c = corners(box) @ camera_from_world[:3,:3].T + camera_from_world[:3,3]
    segments = []
    for a,b in EDGES:
        p,q = c[a].copy(), c[b].copy()
        if p[2] < 0.1 and q[2] < 0.1:
            continue
        if p[2] < 0.1:
            p += (q-p)*(0.1-p[2])/(q[2]-p[2])
        if q[2] < 0.1:
            q += (p-q)*(0.1-q[2])/(p[2]-q[2])
        u = intrinsics @ p; v = intrinsics @ q
        segments.append((u[:2]/u[2], v[:2]/v[2]))
    return segments


class SensorStore:
    def __init__(self, root, lookup):
        self.root = Path(root)
        self.lookup = lookup

    @lru_cache(maxsize=32)
    def lidar(self, token):
        sample = self.lookup['sample'][token]
        sd = self.lookup['sample_data'][sample['data']['LIDAR_TOP']]
        points = np.fromfile(self.root/sd['filename'], dtype=np.float32).reshape(-1,5)[:,:3]
        matrix = global_from_sensor(sd,self.lookup)
        xyz = points @ matrix[:3,:3].T + matrix[:3,3]
        return xyz, matrix

    def camera(self, token):
        sample = self.lookup['sample'][token]
        sd = self.lookup['sample_data'][sample['data']['CAM_FRONT']]
        cal = self.lookup['calibrated_sensor'][sd['calibrated_sensor_token']]
        with Image.open(self.root/sd['filename']) as image:
            img = image.convert('RGB')
        return img, np.linalg.inv(global_from_sensor(sd,self.lookup)), np.array(cal['camera_intrinsic'])

    def camera_geometry(self, token):
        """Calibration/FOV without reading image pixels (including blackout)."""
        sample=self.lookup['sample'][token]
        sd=self.lookup['sample_data'][sample['data']['CAM_FRONT']]
        cal=self.lookup['calibrated_sensor'][sd['calibrated_sensor_token']]
        return np.linalg.inv(global_from_sensor(sd,self.lookup)),np.array(cal['camera_intrinsic']),\
            (sd['width'],sd['height']),sd['timestamp']*1e-6

    def ego_position(self, token):
        sample = self.lookup['sample'][token]
        sd = self.lookup['sample_data'][sample['data']['LIDAR_TOP']]
        return np.array(self.lookup['ego_pose'][sd['ego_pose_token']]['translation'])


def corrupt_camera(image, mode):
    """Same combined settings as existing simulate_camera_corruption.py, fixed seed."""
    if mode == 'blackout':
        return Image.new('RGB', image.size, '#080d14')
    if mode != 'degraded':
        return image.copy()
    result = ImageEnhance.Brightness(image.filter(ImageFilter.GaussianBlur(3))).enhance(0.55)
    rng = np.random.default_rng(42)
    pixels = np.clip(np.asarray(result,dtype=np.float32) + rng.normal(0,12,(*np.asarray(result).shape,)),0,255).astype(np.uint8)
    result = Image.fromarray(pixels)
    w,h = image.size
    rw,rh = int(w*np.sqrt(.12)), int(h*np.sqrt(.12))
    x,y = (w-rw)//2,(h-rh)//2
    ImageDraw.Draw(result).rectangle((x,y,x+rw-1,y+rh-1),fill=(0,0,0))
    return result


def overlay(image, tracks, camera_from_world, intrinsics):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    w,h = result.size
    for box in tracks:
        color = box.get('display_color','#ffbc66' if box['status']=='predicted' else '#4de2b0')
        display_box=dict(box)
        dt=box.get('camera_lidar_offset_s',0)
        if dt:
            display_box['translation']=(np.array(box['translation'])+np.array([*box.get('velocity',[0,0])[:2],0])*dt).tolist()
        segments = camera_segments(display_box,camera_from_world,intrinsics)
        visible = []
        for a,b in segments:
            # Pillow clips lines to the image. Cap extreme near-plane projections.
            a = np.clip(a,[-w*5,-h*5],[w*6,h*6]);b = np.clip(b,[-w*5,-h*5],[w*6,h*6])
            draw.line((*a,*b),fill=color,width=3)
            if 0 <= a[0] < w and 0 <= a[1] < h:
                visible.append(a)
        if visible:
            pos = min(visible,key=lambda p:p[1])
            label = box['tracking_id'].split(':')[-2:]
            draw.text((pos[0],max(0,pos[1]-15)),':'.join(label),fill=color,stroke_width=1,stroke_fill='#111111')
    return result


def overlay_observations(image, detections, label='YOLO'):
    result=image.copy();draw=ImageDraw.Draw(result)
    for d in detections:
        x1,y1,x2,y2=d['xyxy']
        draw.rectangle((x1,y1,x2,y2),outline='#ef9df4',width=2)
        draw.text((x1,max(0,y1-13)),f'{label} {d["class_name"]} {d["confidence"]:.2f}',fill='#ef9df4',stroke_width=1,stroke_fill='#111111')
    return result


def image_uri(image, width=960):
    out = image.copy()
    out.thumbnail((width,600))
    buf = BytesIO();out.save(buf,format='JPEG',quality=85)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')
