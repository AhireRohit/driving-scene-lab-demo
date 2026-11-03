"""Local replay dashboard for frozen learned detections and actual SimpleTrack outputs."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ClientsideFunction, no_update, dcc, html

from .detections import read_mini
from .sensors import SensorStore, corners, EDGES, corrupt_camera, image_uri, overlay, overlay_observations
from .replay_snapshots import ScenarioStore,CONDITION_MAP
from .evidence import COLORS
from .storage import read_json

GREEN = '#4de2b0'
AMBER = '#ffbc66'


class Replay:
    def __init__(self, root, output, demo=False, recorded_bundle=None):
        self.demo=False
        if recorded_bundle is not None:
            if demo:raise ValueError('Recorded replay cannot use procedural mode')
            from .portable import initialize_recorded
            initialize_recorded(self,root,recorded_bundle)
            return
        if demo:
            from .demo import initialize_demo
            initialize_demo(self,root)
            return
        self.root = Path(root)
        self.output = Path(output)
        self.snapshots=ScenarioStore(output) if (self.output/'evidence_manifest.json').exists() else None
        baseline=self.root/'outputs/offline_tracking' if self.snapshots else self.output
        self.tracks = read_json(baseline/'simpletrack.json')
        self.coverage = read_json(baseline/'coverage.json')
        self.evaluation = read_json(baseline/'evaluation.json')
        self.benchmark=read_json(self.output/'benchmark_evaluation.json') if (self.output/'benchmark_evaluation.json').exists() else None
        if self.snapshots and (not self.benchmark or self.benchmark['selection_fingerprint']!=self.snapshots.manifest['selection']['selection_fingerprint']):
            raise ValueError('Missing/stale evidence evaluation. Run scripts/run_sensor_benchmark.py evaluate before launching this output.')
        _, self.lookup = read_mini(self.root/'data/raw/nuscenes')
        self.sensors = SensorStore(self.root/'data/raw/nuscenes', self.lookup)

    @lru_cache(maxsize=128)
    def cloud(self, token):
        xyz, matrix = self.sensors.lidar(token)
        # Only display is downsampled; inference/evaluation never reads this array.
        step = max(1,int(np.ceil(len(xyz)/6500)))
        return xyz[::step], matrix

    def get_frame(self, scene_name, index, experiment, duration=4, method='lidar'):
        if self.snapshots:return self.snapshots.frame(scene_name,index,experiment,duration,method)
        scene = self.tracks['scenes'][scene_name]
        index = min(max(int(index or 0),0),len(scene['tokens'])-1)
        token = scene['tokens'][index]
        variant = 'detection_gap' if experiment=='gap' else 'normal'
        return scene, index, token, scene[variant][token], variant

    def figure(self, scene, index, variant, tracks, selected, map_on, top_view, show_ids=False):
        tokens = scene['tokens']
        xyz,matrix = self.cloud(tokens[index])
        origin = self.cloud(tokens[0])[1][:3,3]
        # Rendering coordinates only: retain every sampled point, halve transfer size.
        xyz = (xyz-origin).astype(np.float32)
        fig = go.Figure()
        if map_on:
            # Past-only accumulation, no annotation masking and no future frames.
            past = [self.cloud(t)[0][::6]-origin for t in tokens[max(0,index-11):index]]
            if past:
                bg = np.vstack(past).astype(np.float32)
                fig.add_trace(go.Scatter3d(x=bg[:,0],y=bg[:,1],z=bg[:,2],mode='markers',
                    marker={'size':1,'color':'#344858','opacity':.35},name='Past scans',hoverinfo='skip'))
        fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',
            marker={'size':1.3,'color':xyz[:,2],'colorscale':[[0,'#213746'],[.45,'#7690a1'],[1,'#e0edf4']],
                    'cmin':-3,'cmax':6,'opacity':.65},name='Current LiDAR',hoverinfo='skip'))
        for state,color in COLORS.items():
            lines = [[],[],[]]
            subset = [t for t in tracks if t.get('evidence_state','LIDAR_ONLY' if t['status']=='detected' else 'PREDICTION_ONLY')==state]
            for box in subset:
                c = corners(box)-origin
                for a,b in EDGES:
                    for axis in range(3):
                        lines[axis].extend([float(c[a,axis]),float(c[b,axis]),None])
            fig.add_trace(go.Scatter3d(x=lines[0],y=lines[1],z=lines[2],mode='lines',
                line={'color':color,'width':3},name=state.replace('_',' ').title(),hoverinfo='skip'))
            if subset:
                c = np.array([t['translation'] for t in subset])-origin
                fig.add_trace(go.Scatter3d(x=c[:,0],y=c[:,1],z=c[:,2],mode='markers+text' if show_ids else 'markers',
                    marker={'size':3,'color':color},customdata=[[t['tracking_id'],t['tracking_name'],t['status']] for t in subset],
                    text=[':'.join(t['tracking_id'].split(':')[-2:]) for t in subset],textfont={'size':9,'color':color},textposition='top center',
                    hovertemplate='%{customdata[0]}<br>%{customdata[1]} · %{customdata[2]}<extra></extra>',showlegend=False))
        visible = {t['tracking_id'] for t in tracks}
        history = {}
        for token in tokens[max(0,index-8):index+1]:
            for t in scene[variant][token]['tracks']:
                if t['tracking_id'] in visible and (not selected or t['tracking_id']==selected):
                    history.setdefault(t['tracking_id'],[]).append(np.array(t['translation'])-origin)
        for tid, positions in history.items():
            if len(positions)>1:
                p = np.array(positions)
                fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='lines',
                    line={'color':'#7cc1fa' if tid==selected else '#547d91','width':4 if tid==selected else 2},
                    showlegend=False,hoverinfo='skip'))
        ego = np.array([self.sensors.ego_position(t)-origin for t in tokens[:index+1]])
        fig.add_trace(go.Scatter3d(x=ego[:,0],y=ego[:,1],z=ego[:,2],mode='lines',line={'color':'#e4edfa','width':4},name='Ego path'))
        here = self.sensors.ego_position(tokens[index])-origin
        fig.add_trace(go.Scatter3d(x=[here[0]],y=[here[1]],z=[here[2]],mode='markers',marker={'size':7,'color':'#ffffff','symbol':'diamond'},name='Ego vehicle'))
        center = here
        axes = {'backgroundcolor':'#101923','gridcolor':'#25313b','zerolinecolor':'#3b4854','color':'#8396a5','showbackground':True}
        fig.update_layout(template='plotly_dark',paper_bgcolor='#101923',plot_bgcolor='#101923',
            margin={'l':0,'r':0,'t':8,'b':0},height=565,legend={'orientation':'h','y':1.02,'font':{'size':10}},
            scene={'aspectmode':'data','xaxis':{**axes,'title':'East offset (m)','range':[center[0]-45,center[0]+45]},
                   'yaxis':{**axes,'title':'North offset (m)','range':[center[1]-45,center[1]+45]},
                   'zaxis':{**axes,'title':'Height offset (m)','range':[-4,12]},
                   'camera':{'eye':{'x':0,'y':0,'z':2.5},'up':{'x':0,'y':1,'z':0}} if top_view else {'eye':{'x':1.2,'y':-1.5,'z':.9}}},
            uirevision=f'{scene["scene_token"]}-{top_view}')
        return fig


def create_app(root, output=None, demo=False, recorded_bundle=None):
    replay = Replay(root, output, demo=demo,recorded_bundle=recorded_bundle)
    app = Dash(__name__, assets_folder=str(Path(root)/'assets'), title='Driving Scene Lab — 3D Tracking & Sensor Failure Analysis')
    scenes = sorted(replay.tracks['scenes'])
    default = 'scene-0916' if 'scene-0916' in scenes else scenes[0]
    covered = replay.coverage['mini_covered']
    app.layout = html.Div([
        dcc.Store(id='playing',data=False), dcc.Interval(id='clock',interval=650,disabled=True),
        dcc.Store(id='frame-rendered'),
        dcc.Store(id='scene-counts',data={name:len(replay.tracks['scenes'][name]['tokens']) for name in scenes}),
        html.Header([html.Div([html.Span('D',className='monogram'),html.Div([html.H1('Driving Scene Lab'),html.P('3D Tracking & Sensor Failure Analysis')])]),
                     html.Div([html.Span(className='live-dot'),'PROCEDURAL DEMO' if replay.demo else 'CACHED RESULTS',html.Small(f'{len(scenes)} illustrative scenes' if replay.demo else f'{len(scenes)} scenes · {covered}/404 frames')],className='header-state')]),
        html.Div([
            html.Aside([
                html.P('REPLAY CONFIGURATION',className='eyebrow'),html.Label('Driving scene'),
                dcc.Dropdown(id='scene',options=[{'label':s,'value':s} for s in scenes],value=default,clearable=False),
                html.P(id='scene-description',className='muted'),
                html.Label('Experiment'),dcc.RadioItems(id='experiment',options=[{'label':'Normal sensors','value':'clean'},
                    {'label':'Camera degraded','value':'degraded'},{'label':'Camera blackout','value':'blackout'},
                    {'label':'LiDAR detection withholding','value':'gap'},{'label':'Both unavailable','value':'both','disabled':not bool(replay.snapshots)}],value='clean',className='radio-group'),
                html.Label('Outage duration'),dcc.Dropdown(id='duration',options=[{'label':f'{d} frames / ≈{d*.5:g}s','value':d} for d in (1,2,4)],value=4,clearable=False,disabled=not bool(replay.snapshots)),
                html.Label('Output policy'),dcc.Dropdown(id='method',options=[{'label':'Scripted track outputs' if replay.demo else 'LiDAR tracker','value':'lidar'},{'label':'Camera confirmation (experimental)','value':'policy'},{'label':'Carry-only illustration' if replay.demo else 'Carry-only baseline','value':'carry'}],value='lidar',clearable=False,disabled=not bool(replay.snapshots)),
                html.Div(id='experiment-note',className='note'),
                html.Label('Display confidence'),dcc.Slider(id='confidence',min=.1,max=.9,step=.05,value=.25,marks={.1:'0.1',.5:'0.5',.9:'0.9'}),
                html.P('Filters the view only. Prediction-only tracks use their last detection score.',className='micro'),
                dcc.Checklist(id='layers',options=[{'label':' Accumulate past scans','value':'map'},{'label':' Bird’s-eye camera','value':'top'},{'label':' Track IDs','value':'ids'}],value=['map','ids'],className='layers'),
                html.Label('Inspect track'),dcc.Dropdown(id='track',options=[],value=None,placeholder='All tracks',clearable=True),
                html.Div([html.Div([html.Span(className='legend-dot',style={'background':color}),state.replace('_',' ').title()]) for state,color in COLORS.items()],className='legend'),
            ]),
            html.Main([
                html.Div([html.Div([html.P('SCENE EXPLORER',className='eyebrow'),html.H2(id='frame-title')]),html.Div(id='frame-badge',className='badge')],className='panel-heading'),
                html.Div(id='stats',className='stats'),
                html.Div([html.Section([dcc.Graph(id='world',config={'displayModeBar':False,'scrollZoom':True}),
                    html.Div('Procedural point cloud and scripted tracks; no recorded sensor data or model accuracy claim.' if replay.demo else 'Display LiDAR ≤6,500 points/scan; analysis data unchanged. Global offsets from first LiDAR origin. Past scans are unmasked context, not a static map.',className='caption')],className='world-panel'),
                    html.Section([html.Div([html.Span('CAMERA SCHEMATIC' if replay.demo else 'CAM_FRONT'),html.Span('Scripted track boxes' if replay.demo else 'Projected LiDAR track boxes',className='muted')],className='camera-heading'),
                        html.Img(id='camera',className='camera-image',alt='Procedural camera illustration' if replay.demo else 'Current front camera with projected learned track boxes'),
                        html.Div(id='camera-caption',className='caption'),
                        html.Div([html.Span('CLEAN CAMERA REFERENCE'),html.Span('Same timestamp',className='muted')],className='camera-heading'),
                        html.Img(id='reference',className='camera-image',alt='Unmodified front camera reference without inference overlays'),
                        html.Div(id='track-detail',className='track-detail')],className='camera-panel')],className='sensor-grid'),
                html.Div([html.Button('▶ Play',id='play',n_clicks=0),html.Button('↤ Start',id='reset',n_clicks=0),
                    html.Div(dcc.Slider(id='frame',min=0,max=40,step=1,value=0,marks={0:'0',10:'10',20:'20',30:'30',40:'40'}),className='timeline'),html.Span(id='timecode',className='timecode')],className='transport'),
                html.Div([html.Section([html.H3('TRACKS AT THIS FRAME'),html.Div(id='track-table')],className='table-panel'),
                    html.Section([html.H3('DEMO NOTES' if replay.demo else 'POST-HOC EVALUATION'),html.Div(id='metrics'),html.P('This public fixture illustrates the interface; recorded-scene evaluation is separate.' if replay.demo else 'Custom same-class XY center matching within 2 m, range ≤50 m. GT objects require a LiDAR return. Not official nuScenes scores.',className='micro'),
                        html.Details([html.Summary('Earlier research & annotation views'),html.P('These retained views predate the learned tracker. Scene-0061 mapping and track identities use annotations.'),
                            html.A('Open archived 4D scene (GT)',href='/archive/animated_4d_scene.html',target='_blank'),html.Br(),
                            html.A('Open archived static map (GT masks)',href='/archive/static_map_scene-0061.html',target='_blank'),html.Br(),
                            html.A('Earlier camera benchmark',href='/archive/camera_robustness_multiframe.png',target='_blank')]) if not replay.demo and recorded_bundle is None else None],className='metrics-panel')],className='bottom-grid'),
            ])
        ],className='shell')
    ])
    from flask import send_from_directory, abort
    @app.server.get('/data-license')
    def data_license():
        if recorded_bundle is None:abort(404)
        from flask import Response
        return Response((Path(recorded_bundle)/'DATA_LICENSE.md').read_text(encoding='utf-8'),mimetype='text/plain')
    @app.server.route('/archive/<path:filename>')
    def archive(filename):
        allowed = {'animated_4d_scene.html','static_map_scene-0061.html','camera_robustness_multiframe.png'}
        if replay.demo or recorded_bundle is not None or filename not in allowed:
            abort(404)
        return send_from_directory(Path(root)/'outputs/visualizations',filename)

    # Keep transport local: a slow frame response must never queue more frames
    # or delay Pause/scene selection behind a server roundtrip.
    app.clientside_callback(ClientsideFunction(namespace='replay',function_name='transport'),
                  Output('frame','value'),Output('frame','max'),Output('frame','marks'),Output('clock','disabled'),Output('play','children'),Output('playing','data'),
                  Input('scene','value'),Input('clock','n_intervals'),Input('play','n_clicks'),Input('reset','n_clicks'),
                  Input('experiment','value'),Input('duration','value'),Input('method','value'),
                  State('frame','value'),State('playing','data'),State('frame-rendered','data'),State('scene-counts','data'))

    @app.callback(Output('track','options'),Output('track','value'),Input('scene','value'),Input('experiment','value'),Input('duration','value'),Input('method','value'),State('track','value'))
    def track_options(scene,experiment,duration,method,selected):
        if replay.snapshots:
            current=replay.snapshots.scene(scene,experiment,duration,method)
            ids=sorted({p['tracking_id'] for f in current['normal'].values() for p in f['all_tracks']})
        else:
            ids=sorted({p['tracking_id'] for variant in ('normal','detection_gap') for f in replay.tracks['scenes'][scene][variant].values() for p in f['tracks']})
        return [{'label':':'.join(t.split(':')[-2:]),'value':t} for t in ids],no_update if selected is None or selected in ids else None

    @app.callback(Output('world','figure'),Output('camera','src'),Output('reference','src'),Output('frame-title','children'),Output('frame-badge','children'),
                  Output('stats','children'),Output('timecode','children'),Output('scene-description','children'),Output('experiment-note','children'),
                  Output('camera-caption','children'),Output('track-table','children'),Output('track-detail','children'),Output('metrics','children'),Output('frame-rendered','data'),
                  Input('scene','value'),Input('frame','value'),Input('experiment','value'),Input('confidence','value'),Input('layers','value'),Input('track','value'),Input('duration','value'),Input('method','value'))
    def update(scene_name,index,experiment,confidence,layers,selected,duration,method):
        scene,index,token,frame,variant=replay.get_frame(scene_name,index,experiment,duration,method)
        tracks=[p for p in frame['tracks'] if p.get('last_detection_score',p['tracking_score'])>=confidence]
        figure=replay.figure(scene,index,variant,tracks,selected,'map' in layers,'top' in layers,'ids' in layers)
        clean,camera_from_world,K=replay.sensors.camera(token)
        camera=frame.get('camera')
        mode=camera['mode'] if camera else experiment
        conditioned=replay.sensors.conditioned_camera(token,mode) if hasattr(replay.sensors,'conditioned_camera') else corrupt_camera(clean,mode)
        image=overlay(conditioned,tracks,camera_from_world,K)
        if camera:
            observations=replay.sensors.display_observations(token,camera['detections']) if hasattr(replay.sensors,'display_observations') else camera['detections']
            image=overlay_observations(image,observations,label='Illustration' if replay.demo else 'YOLO')
        camera_caption='Magenta 2D boxes: actual cached YOLO observations from this image condition. 3D boxes: sensor-evidence colors. Clean reference is visualization only; blackout contributes zero camera observations.' if camera else 'Legacy replay: camera pixels do not generate these LiDAR tracks.'
        if recorded_bundle is not None:camera_caption=html.Div([camera_caption,' ',html.A('Data credits & license',href='/data-license',target='_blank')])
        notes={'clean':'Normal inputs: published LiDAR detections and cached clean-image YOLO.',
               'degraded':'During the selected window Cached YOLO observations use blur 3, brightness 0.55, noise sigma 12 and 12% occlusion. Clean observations resume afterward.',
               'blackout':'During the selected window there are no camera observations. Unchanged LiDAR is an independence baseline, not proof of fusion recovery.',
               'gap':'Published LiDAR detections are withheld; the actual point cloud is not corrupted. Camera observations remain available.',
               'both':'During the selected window: no camera observations and withheld LiDAR detections. Both observation streams resume afterward.'}
        note=notes[experiment]
        if replay.demo:
            camera_caption='Procedural road and scripted observations. No dataset imagery, pretrained detections or benchmark claims.'
            demo_notes={'clean':'Normal scripted observations.','degraded':'Scripted camera degradation reduces illustrative observations in the window.',
                        'blackout':'Camera blackout contributes zero observations in the window.',
                        'gap':'Scripted LiDAR detection withholding; the procedural cloud is unchanged.',
                        'both':'Both scripted observation streams are unavailable in the window.'}
            note=f'Illustrative scene; no live inference or measured accuracy. {demo_notes[experiment]} Window: indices 10–{9+int(duration)}; observations resume afterward.'
        if not replay.snapshots:note='Legacy baseline: camera corruption is display-only; no camera observations are associated.'
        if replay.snapshots and not replay.demo:
            chosen=replay.snapshots.manifest['selection']['selected']
            note+=f' Window: indices 10–{9+int(duration)}. LiDAR confidence {chosen["confidence"]}, max age {chosen["max_age"]} frames.'
        if method=='policy':note+=' Camera confirmation filters outputs only; tracker geometry, identities and lifetime remain unchanged.'
        if method=='carry':note+=' Carry baseline holds pre-window boxes, ignores updates during the window, then causally resumes tracking.'
        events=scene['gap_events'] if variant=='detection_gap' else scene['events']
        lost=sum(e['frame']==index and e['event']=='lost' for e in events)
        elapsed=frame['timestamp_s']-scene[variant][scene['tokens'][0]]['timestamp_s']
        statitems=[('LIDAR OBSERVED',sum(p.get('lidar_associated',p['status']=='detected') for p in tracks),GREEN),
                   ('CAMERA OBSERVED',sum(bool(p.get('camera_association')) for p in tracks),'#da9cff'),
                   ('RETIRED',lost,'#c9d7e2'),('POLICY HIDDEN',frame.get('policy_suppressed',0),AMBER)]
        stats=[html.Div([html.Small(label),html.Strong(str(value),style={'color':color})],className='stat') for label,value,color in statitems]
        def age(value):return 'never' if value is None else f'{value:.2f}s'
        head=html.Tr([html.Th(x) for x in ['ID / CLASS','EVIDENCE','DET. SCORE','CAM IoU','LIDAR AGE','CAM AGE']])
        rows=[]
        for t in sorted(tracks,key=lambda p:(p['tracking_name'],p['tracking_id'])):
            association=t.get('camera_association')
            state=t.get('evidence_state',t['status'])
            rows.append(html.Tr([html.Td(':'.join(t['tracking_id'].split(':')[-2:])),html.Td(state.replace('_',' ').lower(),style={'color':COLORS.get(state,AMBER)}),
                                html.Td(f'{t.get("last_detection_score",t["tracking_score"]):.2f}'),html.Td(f'{association["iou"]:.2f}' if association else '—'),
                                html.Td(age(t.get('age_since_lidar_s'))),html.Td(age(t.get('age_since_camera_s')))]))
        table=html.Table([html.Thead(head),html.Tbody(rows)]) if rows else html.P('No output tracks at this frame and display threshold.',className='muted')
        inspected=next((p for p in frame.get('all_tracks',frame['tracks']) if p['tracking_id']==selected),None)
        if inspected:
            t=inspected;a=t.get('camera_association')
            detail=html.Div([html.Strong(':'.join(selected.split(':')[-2:])),html.P(t.get('evidence_state',t['status'])+' / '+t['upstream_state']),
                html.P('Global xyz (m): '+', '.join(f'{x:.2f}' for x in t['translation'])),
                html.P(f'LiDAR age {age(t.get("age_since_lidar_s"))}; camera age {age(t.get("age_since_camera_s"))}'),
                html.P(f'Camera available: {t.get("camera_available","unrecorded")}; potentially in view: {t.get("camera_visible","unrecorded")}; class mapped: {t.get("camera_class_mapped","unrecorded")}'),
                html.P(f'{"Illustrative observation score" if replay.demo else "Camera detector score"} {a["observation_confidence"]:.2f}; association IoU {a["iou"]:.2f}' if a else 'No current camera association; absence outside the field of view is not disagreement.'),
                html.P(t.get('policy_reason','Legacy output')),html.P('Detector scores, association quality and age are different quantities; none is a calibrated safety probability.',className='micro')])
        else:
            retired=next((e for e in reversed(events) if e['tracking_id']==selected and e['frame']<=index and e['event']=='lost'),None)
            message=f'Retired at frame {retired["frame"]}: upstream LiDAR update-age limit. Camera annotation does not extend track lifetime.' if retired else 'Selected track is not born yet, is inactive, or is below the display threshold.'
            if retired and replay.demo:message=f'Scripted retirement at frame {retired["frame"]}; this illustration is not a measured tracker result.'
            detail=html.P(message if selected else 'Select a track to inspect evidence, observation ages and its recent trajectory.',className='muted')
        if replay.demo:
            metricview=[html.P('Procedural demonstration'),html.P('Scripted geometry and observations exercise the same replay, projection, evidence-association and output-policy interface. They are not SimpleTrack/YOLO predictions or accuracy measurements.'),html.P('Recorded-scene benchmark results are documented separately in the repository.')]
        elif replay.benchmark:
            key=f'{scene_name}/{CONDITION_MAP[experiment]}/{int(duration)}'
            m=replay.benchmark['scenarios'][key]['methods'][method]
            cohort=scene['cohort']
            metricview=[html.P(f'{cohort.title()} scene / {method} / {int(duration)}-frame window'),
                        html.Div([html.Strong(f'{m["outage"]["matched"]} / {m["outage"]["gt_boxes"]}'),html.Span(' matched GT observations during window')]),
                        html.P(f'{m["outage"]["unmatched_predictions"]} unmatched outputs; {m["outage"]["missed_gt"]} misses; {m["outage"]["identity_changes"]} identity changes'),
                        html.P(f'Recovery (4 frames): {m["recovery"]["matched"]}/{m["recovery"]["gt_boxes"]} matches; original IDs retained {m["recovery"]["seed_id_retained"]}/{m["recovery"]["seeded_gt_visible"]}'),
                        html.P(f'Stale (>1s without sensor support): {m["outage"]["stale_prediction_boxes"]} outputs in window.'),
                        html.P('Parameters selected using 0103/0553; evaluation scenes 0796/0916 were previously inspected. This is not untouched held-out validation.',className='micro')]
        else:
            m=replay.evaluation['scenes'][scene_name]['simpletrack_lifecycle']
            metricview=[html.P('Legacy baseline'),html.P(f'{m["matched"]}/{m["gt_boxes"]} matched observations; {m["unmatched_predictions"]} unmatched predictions.')]
        badge=frame['input_status'].replace('_',' ')
        if camera:badge+=f' / camera {mode}'
        rendered={'scene':scene_name,'frame':index,'experiment':experiment,'duration':int(duration),'method':method}
        return figure,image_uri(image),image_uri(clean),f'{scene_name} / frame {index:02d}',badge,stats,f'+{elapsed:.2f}s',scene['description'],note,camera_caption,table,detail,metricview,rendered
    return app
