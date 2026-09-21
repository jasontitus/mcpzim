"""Offline checkpoint timing scenarios and conditional GSQ spool capacity.

No cloud/model imports or resource actions. Supply terminal diagnostics, a
checkpoint receipt/manifest, and explicit JSON assumptions. Never a full ETA.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

GIB = 1024**3
MAX_JSON_BYTES = 32 * 1024**2


def number(value, name, *, integer=False, minimum=0):
    if (type(value) not in ((int,) if integer else (int,float)) or not math.isfinite(value)
            or value < minimum):
        raise ValueError(f'{name} must be a finite {"integer" if integer else "number"} >= {minimum}')
    return value


def read(path):
    with Path(path).open('rb') as stream: data=stream.read(MAX_JSON_BYTES+1)
    if len(data)>MAX_JSON_BYTES:raise ValueError('Input exceeds 32 MiB')
    return json.loads(data)


def checkpoint_manifest(value):
    if not isinstance(value,dict):raise ValueError('Expected checkpoint object')
    manifest=value.get('receipt',{}).get('manifest',value.get('manifest',value))
    if not isinstance(manifest.get('payloads'),dict):raise ValueError('Checkpoint payload manifest required')
    payloads=manifest['payloads']
    if not {'solver','optimizer','scheduler','rng','progress','configuration'}<=set(payloads):
        raise ValueError('Missing checkpoint roles')
    if len(payloads)>256:raise ValueError('Too many checkpoint roles')
    for role,item in payloads.items():
        if not isinstance(item,dict):raise ValueError('Invalid payload descriptor')
        number(item.get('bytes'),role+' bytes',integer=True)
    return manifest


def observed_timings(diagnostics,invocations):
    if not isinstance(diagnostics,dict):raise ValueError('Expected terminal diagnostics map')
    rows=[]; count=0
    for path,report in diagnostics.items():
        if Path(path).name!='performance-report.json':continue
        if not isinstance(report,dict) or report.get('schema')!=1 or not isinstance(report.get('groups'),dict):
            raise ValueError('Unsupported performance report')
        for value in report['groups'].values():
            count+=1
            if count>20000:raise ValueError('Too many timing groups')
            stage=value.get('stage');operation=value.get('operation');status=value.get('status')
            if not isinstance(stage,str) or not isinstance(operation,str) or status not in ('completed','failed'):
                raise ValueError('Invalid timing identity/status')
            samples=number(value.get('count'),'timing count',integer=True,minimum=1)
            seconds=number(value.get('seconds'),'timing seconds')
            dimensions=value.get('dimensions',{})
            if not isinstance(dimensions,dict):raise ValueError('Invalid timing dimensions')
            smoke=stage.startswith('smoke') or any('smoke' in part for part in Path(path).parts[:-1])
            row={'source':path,'stage':stage,'operation':operation,'status':status,
                 'scope':'smoke' if smoke else 'production','count':samples,'seconds':seconds,
                 'block':dimensions.get('block'),'kind':dimensions.get('kind')}
            for key in ('min_seconds','max_seconds'):
                if key in value:row[key]=number(value[key],key)
            rows.append(row)
    blocks=defaultdict(lambda:{'updates':0,'update_seconds':0.,'cache_seconds':0.,'cache_passes':0,'kinds':set()})
    for row in rows:
        if row['scope']!='production' or row['status']!='completed' or row['stage']!='gsq':continue
        if row['operation'] not in ('update_with_input_load','cache_propagation'):continue
        block=number(row['block'],'GSQ block',integer=True)
        if block>=64 or row['kind'] not in ('linear_attention','full_attention'):raise ValueError('Invalid block/type')
        group=blocks[block];group['kinds'].add(row['kind'])
        if len(group['kinds'])!=1:raise ValueError('Block attention type changed')
        if row['operation']=='update_with_input_load':
            group['updates']+=row['count'];group['update_seconds']+=row['seconds']
        else:group['cache_passes']+=row['count'];group['cache_seconds']+=row['seconds']
    coverage=[]
    for block,item in sorted(blocks.items()):
        coverage.append({'block':block,'kind':next(iter(item['kinds'])),
            **{k:v for k,v in item.items() if k!='kinds'},
            'one_epoch_update_pass_measured':item['updates']==invocations,
            'one_cache_pass_measured':item['cache_passes']==1})
    return {'groups':rows,'gsq_blocks':coverage,
            'checkpoint_observations':[v for v in rows if v['operation'] in ('checkpoint_total','checkpoint_publish','checkpoint_restore')],
            'nested_accounting':'checkpoint_total encloses publication; stage_total encloses components. Do not add these parents to children.'}


def scenario(value,fields,name):
    if value is None:return None
    if not isinstance(value,dict) or not isinstance(value.get('source'),str) or not value['source'].strip():
        raise ValueError(name+' requires an explicit source/assumption description')
    for field in fields:number(value.get(field),name+'.'+field)
    return value


def plan(diagnostics,checkpoint,assumptions):
    if not isinstance(assumptions,dict):raise ValueError('Explicit assumptions object required')
    manifest=checkpoint_manifest(checkpoint);payloads=manifest['payloads']
    total=number(assumptions.get('total_blocks',64),'total_blocks',integer=True,minimum=1)
    invocations=number(assumptions.get('invocations',87),'invocations',integer=True,minimum=1)
    epochs=number(assumptions.get('gsq_epochs',1),'gsq_epochs',integer=True,minimum=1)
    if total>64:raise ValueError('Planner supports at most 64 Qwen blocks')
    completed=sorted(int(m.group(1)) for name in payloads if (m:=re.fullmatch(r'candidate_block_(\d+)',name)))
    if completed!=list(range(len(completed))) or len(completed)>total:
        raise ValueError('Completed candidate blocks must form a contiguous prefix')
    remaining=total-len(completed)
    observations=observed_timings(diagnostics,invocations)
    synchronous=scenario(assumptions.get('synchronous'),['seconds_per_block'],'synchronous')
    overlap=scenario(assumptions.get('overlap'),['producer_seconds_per_block','uploader_seconds_per_snapshot',
        'background_snapshots','initial_upload_backlog_seconds','drain_seconds','restore_seconds'],'overlap')
    rate=assumptions.get('compute_usd_per_hour')
    if rate is not None:number(rate,'compute_usd_per_hour')
    scenarios={}
    if synchronous is not None:
        seconds=remaining*synchronous['seconds_per_block']
        scenarios['current_sync_gsq']={'seconds':seconds,'hours':seconds/3600,
            'compute_usd':seconds/3600*rate if rate is not None else None,
            'source':synchronous['source'],'remaining_blocks':remaining,
            'scope':'Conditional GSQ block work only; transfers the supplied measured/assumed block interval unchanged.',
            'excludes':['initialization/restore','embedding','head','RCO','packing','evaluation','future interruptions']}
    if overlap is not None:
        snapshots=number(overlap['background_snapshots'],'background_snapshots',integer=True)
        producer=remaining*overlap['producer_seconds_per_block']
        uploader=snapshots*overlap['uploader_seconds_per_snapshot']+overlap['initial_upload_backlog_seconds']
        seconds=max(producer,uploader)+overlap['drain_seconds']+overlap['restore_seconds']
        scenarios['overlapped_gsq']={'seconds':seconds,'hours':seconds/3600,
            'compute_usd':seconds/3600*rate if rate is not None else None,
            'producer_seconds':producer,'background_uploader_seconds':uploader,
            'drain_seconds':overlap['drain_seconds'],'restore_seconds':overlap['restore_seconds'],
            'source':overlap['source'], 'formula':'max(producer, background uploader) + final drain + restore',
            'scope':'Conditional overlap model, not measured achieved speed or an upper/lower bound.',
            'assumptions':['Producer includes synchronous serialization/local copying, all checkpoints and block work.',
                'Background upload volume excludes the separately counted final drain; backlog includes only unfinished upload work.',
                'The max model assumes sustained overlap and compatible release times. Startup/starvation, CPU/disk contention and retries may make it inaccurate.',
                'Coalescing reduces uploads, not local publications or retained local history.']}
    retained=assumptions.get('retention',{'strategy':'retain_all'})
    if not isinstance(retained,dict) or retained.get('strategy') not in ('retain_all','quiescent_windows'):
        raise ValueError('Use retain_all or quiescent_windows; concurrent GC is unsupported')
    window=remaining
    if retained['strategy']=='quiescent_windows':
        window=number(retained.get('blocks_per_window'),'blocks_per_window',integer=True,minimum=1)
        window=min(window,remaining)
        if retained.get('producer_stopped_during_gc') is not True:
            raise ValueError('Quiescent retention requires the producer to be stopped during GC')
    attempt_seconds=assumptions.get('attempt_seconds')
    cadence=assumptions.get('checkpoint_seconds')
    if (attempt_seconds is None)!=(cadence is None):
        raise ValueError('Specify attempt_seconds and pinned checkpoint_seconds together')
    if attempt_seconds is not None:
        number(attempt_seconds,'attempt_seconds',minimum=1)
        number(cadence,'checkpoint_seconds',minimum=30)
    periodic_limit=window*epochs*invocations
    if attempt_seconds is not None:periodic_limit=min(periodic_limit,math.ceil(attempt_seconds/cadence))
    cursor=re.match(r'gsq-b(\d+)-',manifest.get('snapshot',''))
    terminal_resume=2 if cursor and int(cursor.group(1))<len(completed) else 0
    snapshot_limit=periodic_limit+2*window+(1 if remaining else 0)+terminal_resume
    measured_state=sum(payloads[r]['bytes'] for r in ('solver','optimizer','scheduler','rng','progress'))
    measured_cache=max([v['bytes'] for k,v in payloads.items() if k.startswith('cache_')]+[0])
    measured_candidate=max([v['bytes'] for k,v in payloads.items() if k.startswith('candidate_block_')]+[0])
    payload_bytes=sum(v['bytes'] for v in payloads.values())
    recovery=6*payload_bytes+10*GIB
    bounds=assumptions.get('disk_bounds')
    disk={'scope':'Conditional additional free-space requirement for remaining GSQ, not the complete quantization pipeline.',
        'strategy':retained['strategy'],'history_window_blocks':window,
        'existing_checkpoint_payload_bytes':payload_bytes,'recovery_admission_bytes':recovery,
        'observed_state_bytes':measured_state,'observed_cache_bytes':measured_cache,
        'observed_candidate_block_bytes':measured_candidate,
        'required_free_bytes':None,'headroom':'unknown_missing_explicit_bounds',
        'snapshot_bound_per_block_without_time_cap':epochs*invocations+2,
        'periodic_snapshot_upper_count':periodic_limit,'forced_block_snapshot_upper_count':2*window,
        'forced_stop_snapshot_allowance':1 if remaining else 0,
        'terminal_resume_snapshot_allowance':terminal_resume,
        'attempt_seconds':attempt_seconds,'checkpoint_seconds':cadence,
        'notes':['Snapshots can be published after every update plus forced block start/end. Cadence alone does not prove a tighter count.',
            'The optional wall-duration/cadence publication cap assumes a correctly enforced single-attempt deadline and the declared unchanged solver cadence.',
            'Only complete verified local manifests are GC roots. Do not delete apparently unreferenced objects while the producer is active: object installation precedes commit publication.',
            'Use actual current free space: existing retained history is already consuming disk and cannot be assumed reclaimed.',
            'Final candidate archives accumulate even when old active-state snapshots are reclaimed.',
            'Quiescent-window GC is a hypothetical policy here; the current bridge implements no GC. Window boundaries and retained roots must be enforced separately.',
            'Repeated interruptions/new attempts need a fresh capacity check; this is one configured attempt/window model.',
            'Head/RCO state, final packing, retained old stage stores and cross-stage duplicate objects require separate capacity planning.']}
    if bounds is not None:
        if not isinstance(bounds,dict) or not isinstance(bounds.get('source'),str) or not bounds['source'].strip():
            raise ValueError('disk_bounds needs an explicit source/assumption description')
        state=number(bounds.get('state_bytes_per_snapshot'),'state_bytes_per_snapshot',integer=True,minimum=measured_state)
        cache=number(bounds.get('cache_bytes_per_block'),'cache_bytes_per_block',integer=True,minimum=measured_cache)
        candidate=number(bounds.get('candidate_bytes_per_block'),'candidate_bytes_per_block',integer=True,minimum=measured_candidate)
        free=number(bounds.get('available_free_bytes'),'available_free_bytes',integer=True)
        extra=number(bounds.get('additional_reserve_bytes',0),'additional_reserve_bytes',integer=True)
        # Never assume sampled block-2 empty Adam bounds a trained block or that
        # upload coalescing bounds local publication frequency.
        snapshots=snapshot_limit
        metadata_per_snapshot=number(bounds.get('metadata_bytes_per_snapshot',8*1024**2),
            'metadata_bytes_per_snapshot',integer=True,minimum=8*1024**2)
        historical_state=snapshots*state
        historical_cache=window*cache
        candidates=remaining*candidate
        # Independent conservative scratch allowance, intentionally not claimed
        # as exact peak: serialization, upload bundle, copy temp, two live caches.
        transient=3*state+2*cache+max(v['bytes'] for v in payloads.values())
        anchors=0
        if retained['strategy']=='quiescent_windows':
            pins=number(retained.get('pinned_snapshots',2),'pinned_snapshots',integer=True,minimum=2)
            anchors=pins*(state+cache)
        metadata=snapshots*metadata_per_snapshot
        required=recovery+historical_state+historical_cache+candidates+transient+anchors+metadata+extra
        disk.update(required_free_bytes=required,available_free_bytes=free,
            free_space_margin_bytes=free-required,
            headroom='insufficient_under_declared_bounds' if free<required else 'conditional_gsq_headroom_only',
            bound_source=bounds['source'],new_state_history_bytes=historical_state,
            new_cache_history_bytes=historical_cache,new_candidate_archive_bytes=candidates,
            transient_allowance_bytes=transient,pinned_anchor_bytes=anchors,additional_reserve_bytes=extra,
            snapshot_metadata_allowance_bytes=metadata,
            maximum_new_local_snapshots=snapshots,
            upper_bounds_independently_verified=False)
    return {'schema_version':1,'status':'offline_conditional_checkpoint_plan',
        'workload':{'total_blocks':total,'completed_candidate_blocks':completed,'remaining_blocks':remaining,
                    'invocations':invocations,'gsq_epochs':epochs,'remaining_update_upper_count':remaining*epochs*invocations},
        'observations':observations,'timing_scenarios':scenarios,'disk':disk,
        'complete_job_eta_seconds':None,'full_pipeline_disk_admitted':False,
        'remote_checkpoint_verified':False,
        'missing_measurements':['Full-attention production corpus and later-block behavior',
            'Producer timings with actual retained-disk checkpoint copies and concurrent uploader',
            'Uploader throughput/backlog/drain under actual contention and coalescing',
            'Learned-head updates/checkpoints, production RCO, packing and evaluation'],
        'cost_scope':'Scenario VM compute only at caller-supplied rate; no current-price verification, storage/network/preparation costs or invoice claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostics','checkpoint','assumptions'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=plan(read(args.diagnostics),read(args.checkpoint),read(args.assumptions))
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:stream.write(encoded)
    else:print(encoded,end='')


if __name__=='__main__':main()
