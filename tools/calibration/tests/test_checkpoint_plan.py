import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import checkpoint_plan as module


def checkpoint():
    payloads={k:{'bytes':value} for k,value in {'solver':100,'optimizer':200,'scheduler':10,'rng':20,'progress':30,
        'configuration':100,'initial_candidates':1000,'cache_block_2':400,'candidate_block_0':50,'candidate_block_1':50}.items()}
    return {'schema':3,'payloads':payloads,'snapshot':'gsq-b002-s00000174-e000-q00000-p00004'}


def timing(stage,operation,count,seconds,block=None,kind=None,status='completed'):
    return {'stage':stage,'operation':operation,'status':status,'count':count,'seconds':seconds,
            'dimensions':{'block':block,'kind':kind}}


def diagnostics():
    groups={str(i):value for i,value in enumerate([
        timing('gsq','update_with_input_load',87,18,0,'linear_attention'),
        timing('gsq','cache_propagation',1,4,0,'linear_attention'),
        timing('gsq','checkpoint_total',2,200),timing('gsq','checkpoint_publish',2,190),
        timing('gsq','stage_total',1,300),timing('gsq','update_with_input_load',1,99,3,'full_attention',status='failed')])}
    return {'04-gsq/performance-report.json':{'schema':1,'groups':groups},
        '01-smoke_gsq/performance-report.json':{'schema':1,'groups':{'0':timing('smoke_gsq','update',1,1,3,'full_attention')}}}


def assumptions():
    return {'total_blocks':64,'invocations':87,'gsq_epochs':1,'compute_usd_per_hour':2,
        'synchronous':{'seconds_per_block':229,'source':'Observed commit timestamps; conditional transfer across all types.'},
        'overlap':{'producer_seconds_per_block':50,'uploader_seconds_per_snapshot':100,'background_snapshots':10,
                   'initial_upload_backlog_seconds':200,'drain_seconds':150,'restore_seconds':300,
                   'source':'Hypothetical parameters; background excludes final drain.'},
        'disk_bounds':{'state_bytes_per_snapshot':400,'cache_bytes_per_block':450,'candidate_bytes_per_block':60,
                       'available_free_bytes':1000*module.GIB,'source':'Declared upper sizes, not independent proof.'}}


def test_remaining_blocks_sync_and_overlap_are_explicit_scenarios():
    result=module.plan(diagnostics(),checkpoint(),assumptions())
    assert result['workload']['completed_candidate_blocks']==[0,1]
    assert result['workload']['remaining_blocks']==62
    sync=result['timing_scenarios']['current_sync_gsq']
    assert sync['seconds']==62*229
    assert sync['compute_usd']==62*229/3600*2
    overlap=result['timing_scenarios']['overlapped_gsq']
    assert overlap['producer_seconds']==3100 and overlap['background_uploader_seconds']==1200
    assert overlap['seconds']==3100+150+300
    assert result['complete_job_eta_seconds'] is None
    assert result['full_pipeline_disk_admitted'] is False


def test_uploader_bottleneck_and_disjoint_drain():
    a=assumptions();a['overlap']['background_snapshots']=100
    result=module.plan({},checkpoint(),a)['timing_scenarios']['overlapped_gsq']
    assert result['seconds']==10200+150+300


def test_no_nested_totals_or_failed_smoke_coverage_projection():
    result=module.plan(diagnostics(),checkpoint(),{})
    assert result['timing_scenarios']=={}
    blocks=result['observations']['gsq_blocks']
    assert len(blocks)==1 and blocks[0]['block']==0
    assert blocks[0]['update_seconds']==18 and blocks[0]['cache_seconds']==4
    assert blocks[0]['one_epoch_update_pass_measured'] is True
    assert len(result['observations']['checkpoint_observations'])==2
    assert result['disk']['required_free_bytes'] is None


def test_no_gc_accounts_all_local_states_despite_cloud_coalescing():
    a=assumptions();result=module.plan({},checkpoint(),a);disk=result['disk']
    assert disk['strategy']=='retain_all'
    assert disk['maximum_new_local_snapshots']==62*(87+2)+1
    assert disk['new_state_history_bytes']==(62*89+1)*400
    assert disk['new_cache_history_bytes']==62*450
    assert disk['new_candidate_archive_bytes']==62*60
    a['overlap']['background_snapshots']=1
    assert module.plan({},checkpoint(),a)['disk']==disk


def test_quiescent_windows_keep_accumulating_candidates_and_pinned_snapshots():
    a=assumptions();a['retention']={'strategy':'quiescent_windows','blocks_per_window':2,
                                  'producer_stopped_during_gc':True,'pinned_snapshots':2}
    disk=module.plan({},checkpoint(),a)['disk']
    assert disk['maximum_new_local_snapshots']==2*89+1
    assert disk['new_cache_history_bytes']==900
    assert disk['new_candidate_archive_bytes']==62*60
    assert disk['pinned_anchor_bytes']==2*(400+450)


def test_duration_cadence_bound_is_a_count_bound_not_a_throughput_estimate():
    a=assumptions();a.update(attempt_seconds=5*3600,checkpoint_seconds=120)
    disk=module.plan({},checkpoint(),a)['disk']
    assert disk['periodic_snapshot_upper_count']==150
    assert disk['maximum_new_local_snapshots']==150+124+1
    a['attempt_seconds']=120.1
    assert module.plan({},checkpoint(),a)['disk']['periodic_snapshot_upper_count']==2


def test_disk_insufficiency_and_non_guarantee_when_space_fits():
    a=assumptions();a['disk_bounds']['available_free_bytes']=0
    disk=module.plan({},checkpoint(),a)['disk']
    assert disk['headroom']=='insufficient_under_declared_bounds'
    a['disk_bounds']['available_free_bytes']=disk['required_free_bytes']
    disk=module.plan({},checkpoint(),a)['disk']
    assert disk['headroom']=='conditional_gsq_headroom_only'
    assert disk['upper_bounds_independently_verified'] is False
    assert disk['free_space_margin_bytes']==0


@pytest.mark.parametrize('retention',[
 {'strategy':'concurrent_gc'}, {'strategy':'quiescent_windows','blocks_per_window':2},
 {'strategy':'quiescent_windows','blocks_per_window':0,'producer_stopped_during_gc':True}])
def test_unsafe_or_undefined_retention_rejected(retention):
    with pytest.raises(ValueError):module.plan({},checkpoint(),{'retention':retention})


@pytest.mark.parametrize('field,value',[
 ('state_bytes_per_snapshot',359),('cache_bytes_per_block',399),('candidate_bytes_per_block',49),
 ('available_free_bytes',-1),('metadata_bytes_per_snapshot',10)])
def test_bounds_cannot_undercut_observed_sizes(field,value):
    a=assumptions();a['disk_bounds'][field]=value
    with pytest.raises(ValueError):module.plan({},checkpoint(),a)


@pytest.mark.parametrize('value',[True,-1,float('nan'),float('inf')])
def test_invalid_timing_or_counts_rejected(value):
    a=assumptions();a['overlap']['producer_seconds_per_block']=value
    with pytest.raises(ValueError):module.plan({},checkpoint(),a)


def test_missing_source_is_not_accepted_as_measurement():
    a=assumptions();a['synchronous'].pop('source')
    with pytest.raises(ValueError,match='source'):module.plan({},checkpoint(),a)


def test_noncontiguous_completed_blocks_rejected():
    c=checkpoint();c['payloads']['candidate_block_3']=c['payloads'].pop('candidate_block_1')
    with pytest.raises(ValueError,match='contiguous'):module.plan({},c,{})


def test_explicit_zero_remaining_does_not_claim_full_job_done():
    a=assumptions();a['total_blocks']=2
    result=module.plan({},checkpoint(),a)
    assert result['workload']['remaining_blocks']==0
    assert result['disk']['maximum_new_local_snapshots']==0
    assert result['complete_job_eta_seconds'] is None
    assert result['timing_scenarios']['current_sync_gsq']['seconds']==0


def test_wrapper_checkpoint_manifest_and_inputs_unchanged():
    c={'receipt':{'manifest':checkpoint()}};a=assumptions();d=diagnostics();before=copy.deepcopy((c,a,d))
    module.plan(d,c,a)
    assert (c,a,d)==before


def test_cli_no_overwrite(tmp_path):
    paths=[]
    for name,value in [('diagnostics',diagnostics()),('checkpoint',checkpoint()),('assumptions',assumptions())]:
        path=tmp_path/(name+'.json');path.write_text(json.dumps(value));paths.extend(['--'+name,str(path)])
    output=tmp_path/'plan.json';command=[sys.executable,module.__file__,*paths,'--output',str(output)]
    first=subprocess.run(command,capture_output=True,text=True)
    assert first.returncode==0,first.stderr
    assert json.loads(output.read_text())['complete_job_eta_seconds'] is None
    assert subprocess.run(command,capture_output=True).returncode!=0


def test_completed_block_terminal_resume_keeps_republication_allowance():
    c=checkpoint();c['snapshot']='gsq-b001-s00000174-e001-q00000-p00003'
    disk=module.plan({},c,assumptions())['disk']
    assert disk['terminal_resume_snapshot_allowance']==2
    assert disk['maximum_new_local_snapshots']==62*89+1+2
