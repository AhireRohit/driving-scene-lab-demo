"""Staged, reproducible tuning and frozen multimodal failure benchmark."""
from copy import deepcopy
from collections import Counter
import subprocess
import time
from pathlib import Path
import yaml
from threadpoolctl import threadpool_limits
from .detections import read_mini,scene_samples,sha256
from .storage import read_json,write_json,fingerprint
from .tracking import track_scene
from .sensors import SensorStore
from .evidence import annotate_frames,GATES

TUNING=('scene-0103','scene-0553')
EVALUATION=('scene-0796','scene-0916')
GRID=[{'confidence':c,'max_age':a} for c in (.1,.3,.5) for a in (2,5)]
DURATIONS=(1,2,4)
CONDITIONS=('normal','degraded','blackout','lidar_gap','both')


def config_for(root,settings):
    c=yaml.safe_load((Path(root)/'third_party/SimpleTrack/configs/nu_configs/giou.yaml').read_text())
    c['running']['score_threshold']=settings['confidence'];c['running']['max_age_since_update']=settings['max_age']
    c['data_loader']['pc']=False
    return c


def inputs(root):
    root=Path(root)
    coverage=read_json(root/'outputs/offline_tracking/coverage.json')
    payload=read_json(root/'outputs/offline_tracking/mini_detections.json')
    tables,lookup=read_mini(root/'data/raw/nuscenes')
    # Validate source subset against recorded full artifact before experiment reuse.
    if sha256(root/'data/external/centerpoint/mctrack_val.json')!=coverage['sha256']:raise ValueError('Source hash mismatch')
    reproduced=root/'outputs/baseline_reproduced/mini_detections.json'
    if reproduced.exists() and fingerprint(read_json(reproduced))!=fingerprint(payload):
        raise ValueError('Local subset differs from newly reproduced source-validated detections')
    return payload,coverage,tables,lookup


def tune(root,output):
    root=Path(root);output=Path(output)
    payload,coverage,tables,lookup=inputs(root)
    snapshots=[]
    for settings in GRID:
        key=f'c{settings["confidence"]}_a{settings["max_age"]}';path=output/'tuning'/f'{key}.json.gz'
        print('Freeze tuning',key,flush=True)
        runs={};started=time.perf_counter()
        with threadpool_limits(limits=1):
            for scene in tables['scene']:
                if scene['name'] not in TUNING:continue
                samples=list(scene_samples(scene,lookup['sample']))
                normal,events=track_scene(scene,samples,payload['results'],root/'third_party/SimpleTrack',config_for(root,settings),lookup,verbose=False)
                gap,gap_events=track_scene(scene,samples,payload['results'],root/'third_party/SimpleTrack',config_for(root,settings),lookup,gap=(10,14),verbose=False)
                runs[scene['name']]={'normal':normal,'gap':gap,'events':events,'gap_events':gap_events}
        snapshot={'settings':settings,'scenes':runs,'runtime_s':time.perf_counter()-started,'source_sha256':coverage['sha256']}
        write_json(path,snapshot);snapshots.append((path,settings))
    # Every candidate's inference has been persisted before annotation access.
    from .experiment_evaluation import load_truth,summarize,combine
    truth,ego=load_truth(root/'data/raw/nuscenes',TUNING)
    scores=[]
    for path,settings in snapshots:
        snapshot=read_json(path);normal=[];gap=[]
        for s in snapshot['scenes'].values():
            normal.append(summarize(s['normal'],truth,ego))
            g=summarize(s['gap'],truth,ego)
            gap.append(g['outage'])
        n=combine(normal);g=combine(gap)
        score=.75*n['f1']+.25*g['f1']-.02*n['identity_changes']/max(n['gt_boxes'],1)
        scores.append({'settings':settings,'normal':n,'gap_outage':g,'score':score,
                       'runtime_s':snapshot['runtime_s'],'snapshot':str(path.relative_to(output)),
                       'snapshot_sha256':sha256(path)})
    winner=max(scores,key=lambda r:(r['score'],-r['settings']['max_age'],-r['settings']['confidence']))
    result={'selected':winner['settings'],'candidates':scores,'tuning_scenes':TUNING,'evaluation_scenes':EVALUATION,
            'selection_rule':'.75*normal_F1 + .25*4-frame-gap_F1 - .02*normal_identity_changes/GT; ties prefer shorter max age then lower confidence',
            'split_warning':'All four mini scenes were previously inspected; this is a small development split, not untouched held-out generalization.',
            'source_sha256':coverage['sha256'],'grid':GRID,'selected_before_evaluation':True}
    result['selection_fingerprint']=fingerprint(result)
    write_json(output/'selection.json',result)
    print('Selected',result['selected'],flush=True)
    return result


def freeze_benchmark(root,output):
    root=Path(root);output=Path(output)
    selection=read_json(output/'selection.json');camera=read_json(output/'camera_manifest.json')
    payload,coverage,tables,lookup=inputs(root)
    store=SensorStore(root/'data/raw/nuscenes',lookup)
    settings=selection['selected'];config=config_for(root,settings)
    manifest={'schema_version':1,'selection':selection,'source':coverage['source'],'source_sha256':coverage['sha256'],
              'camera':{k:v for k,v in camera.items() if k!='frames'},'association_gates':GATES,
              'mini_subset_sha256':sha256(root/'outputs/offline_tracking/mini_detections.json'),
              'upstream_commit':subprocess.check_output(['git','-C',str(root/'third_party/SimpleTrack'),'rev-parse','HEAD'],text=True).strip(),
              'policy':'Output confirmation only: require a current compatible camera match when potentially in view, mapped, available and timestamp-eligible; otherwise unchanged LiDAR fallback. No geometry/ID/filter updates.',
              'durations_frames':DURATIONS,'conditions':CONDITIONS,'start_index':10,'scenes':{},'files':{}}
    started=time.perf_counter()
    for scene in sorted(tables['scene'],key=lambda s:s['name']):
        if scene['name'] not in TUNING+EVALUATION:continue
        name=scene['name'];samples=list(scene_samples(scene,lookup['sample']))
        tokens=[s['token'] for s in samples]
        manifest['scenes'][name]={'description':scene['description'],'scene_token':scene['token'],'tokens':tokens,
                                'cohort':'tuning' if name in TUNING else 'evaluation'}
        print('Freeze sensor benchmark',name,flush=True)
        with threadpool_limits(limits=1):
            normal,normal_events=track_scene(scene,samples,payload['results'],root/'third_party/SimpleTrack',config,lookup,verbose=False)
            for duration in DURATIONS:
                gap=(10,10+duration)
                missing,missing_events=track_scene(scene,samples,payload['results'],root/'third_party/SimpleTrack',config,lookup,gap=gap,verbose=False)
                carry,carry_events=track_scene(scene,samples,payload['results'],root/'third_party/SimpleTrack',config,lookup,carry_gap=gap,verbose=False)
                for condition in CONDITIONS:
                    raw=missing if condition in ('lidar_gap','both') else normal
                    events=missing_events if condition in ('lidar_gap','both') else normal_events
                    def packet(token,index):
                        mode=('degraded' if condition=='degraded' else 'blackout') if condition in ('degraded','blackout','both') and gap[0]<=index<gap[1] else 'clean'
                        value=read_json(root/'data/cache/camera'/camera['frames'][token][mode])
                        if value['identity']['sample_token']!=token or value['identity']['mode']!=mode:raise ValueError('Wrong camera cache packet')
                        if value['identity']['model_sha256']!=camera['model_sha256'] or value['identity']['settings']!=camera['settings']:
                            raise ValueError('Camera model/settings differ from the manifest')
                        return value
                    annotate_start=time.perf_counter()
                    annotated=annotate_frames(raw,store,packet)
                    carry_annotated=annotate_frames(carry,store,packet,negative_controls=False)
                    for token,row in annotated.items():row['carry_tracks']=carry_annotated[token]['tracks']
                    path=Path('scenarios')/name/f'{condition}_{duration}.json.gz'
                    data={'scene':name,'condition':condition,'duration_frames':duration,'start_index':10,'settings':settings,
                          'source_sha256':coverage['sha256'],'selection_fingerprint':selection['selection_fingerprint'],
                          'frames':annotated,'events':events,'carry_events':carry_events,
                          'annotation_runtime_s':time.perf_counter()-annotate_start,
                          'lidar_runtime_s':sum(f.get('runtime_s',0) for f in raw.values()),
                          'carry_runtime_s':sum(f.get('runtime_s',0) for f in carry.values())}
                    write_json(output/path,data)
                    manifest['files'][f'{name}/{condition}/{duration}']={'path':path.as_posix(),'sha256':sha256(output/path)}
    manifest['freeze_runtime_s']=time.perf_counter()-started
    write_json(output/'evidence_manifest.json',manifest)
    print('All inference snapshots frozen; annotations have not been used in this stage.',flush=True)
    return manifest


def evaluate_benchmark(root,output):
    from .experiment_evaluation import load_truth,summarize,combine
    root=Path(root);output=Path(output);manifest=read_json(output/'evidence_manifest.json')
    # Manifest hashes are checked before the evaluator accesses annotations.
    for item in manifest['files'].values():
        if sha256(output/item['path'])!=item['sha256']:raise ValueError('Frozen scenario hash mismatch')
    truth,ego=load_truth(root/'data/raw/nuscenes',set(manifest['scenes']))
    report={'selection_fingerprint':manifest['selection']['selection_fingerprint'],
            'eligibility':'Seven same-class categories; one-to-one XY distance <=2m; both predictions/GT within 50m of LIDAR_TOP ego pose; GT requires num_lidar_pts>0. Unmatched is not automatically a false positive.',
            'identity':'GT-to-track changes across successive matched observations including gaps, not official IDSW; common pre-outage LiDAR seeds define original-ID continuity.',
            'stale_definition':'No current sensor association and time since latest sensor observation >1s. Camera-only support does not update tracker state.',
            'scenarios':{},'aggregate':{},'by_outage_age':{}}
    aggregate={};age_buckets={}
    for key,item in manifest['files'].items():
        data=read_json(output/item['path']);cohort=manifest['scenes'][data['scene']]['cohort']
        methods={}
        for method in ('lidar','carry','policy'):
            metrics=summarize(data['frames'],truth,ego,duration=data['duration_frames'],method=method)
            methods[method]=metrics
            group=f'{cohort}/{data["condition"]}/{data["duration_frames"]}/{method}'
            aggregate.setdefault(group,[]).append(metrics)
            for f in metrics['per_frame']:
                if f['phase'] in ('outage','recovery'):
                    age_key=group+f'/{f["phase"]}/{f["age_frames"]}'
                    age_buckets.setdefault(age_key,[]).append(f)
        controls={}
        for shift in ('3','5'):
            total=Counter()
            for token,frame in data['frames'].items():
                if not 10<=frame['frame_index']<10+data['duration_frames']:continue
                boxes=frame['negative_controls'][shift]
                total.update({'hypotheses':len(boxes),'camera_visible':sum(b['camera_visible'] for b in boxes),
                              'camera_supported':sum(b['camera_supported'] for b in boxes),
                              'localization_matches':len(__import__('shadowdrive.evaluation',fromlist=['match_centers']).match_centers(boxes,truth[token]))})
            controls[shift]=dict(total)
        report['scenarios'][key]={'methods':methods,'negative_controls_shift_global_x_m':controls,
                                   'lidar_runtime_s':data['lidar_runtime_s'],'carry_runtime_s':data['carry_runtime_s'],
                                   'annotation_runtime_s':data['annotation_runtime_s']}
    for key,rows in aggregate.items():
        report['aggregate'][key]={'whole_scene':combine(rows),'outage':combine([r['outage'] for r in rows]),
                                   'recovery':combine([r['recovery'] for r in rows])}
    for key,rows in age_buckets.items():
        report['by_outage_age'][key]={**combine(rows),'mean_age_s':sum(r['age_s'] for r in rows)/len(rows)}
    write_json(output/'benchmark_evaluation.json',report)
    return report
