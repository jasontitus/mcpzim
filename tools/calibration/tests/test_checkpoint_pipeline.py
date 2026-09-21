import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import continuation as controller


def fixture(tmp_path, producer_codes, upload_codes):
    ledger=tmp_path/'ledger.json';done=tmp_path/'done.json';events=[]
    spec={'argv':['docker','run','producer'],'container_name':'continue-test-gsq',
          'owner_run_id':'zimfo-gpu-123456789abc','owned_container_ledger':str(ledger),
          'log':str(tmp_path/'solver.log'),'producer_done':str(done),
          'publisher':{'argv':['docker','run','publisher'],'container_name':'continue-test-gsq-publisher',
                       'log':str(tmp_path/'publisher.log')}}
    class Process:
        def __init__(self, name, codes):self.name=name;self.codes=iter(codes);self.returncode=None;self.killed=False
        def poll(self):
            self.returncode=next(self.codes,self.returncode)
            return self.returncode
        def wait(self,timeout):return self.returncode
        def kill(self):self.killed=True;self.returncode=-9
    producer=Process('producer',producer_codes);upload=Process('publisher',upload_codes)
    def factory(argv,**kwargs):
        assert json.loads(ledger.read_text())['publisher_container_name']=='continue-test-gsq-publisher'
        events.append(argv[-1]);return upload if argv[-1]=='publisher' else producer
    def stop(argv,**kwargs):events.append(argv);return SimpleNamespace(returncode=0)
    return spec,producer,upload,events,factory,stop,done,ledger


def test_producer_keeps_running_while_publisher_runs_and_final_drain_is_required(tmp_path):
    spec,p,u,events,factory,stop,done,ledger=fixture(tmp_path,[None,None,0],[None,None,None,0])
    pauses=[]
    controller.run_pipeline(spec,1000,process_factory=factory,clock=lambda:0,sleep=pauses.append,stop=stop)
    assert events==['publisher','producer']
    assert json.loads(done.read_text())['producer_returncode']==0
    assert len(pauses)==3 and not ledger.exists()


def test_failed_producer_still_drains_completed_local_checkpoint(tmp_path):
    spec,p,u,events,factory,stop,done,ledger=fixture(tmp_path,[1],[None,0])
    with pytest.raises(controller.subprocess.CalledProcessError):
        controller.run_pipeline(spec,1000,process_factory=factory,clock=lambda:0,sleep=lambda _:None,stop=stop)
    assert json.loads(done.read_text())['producer_returncode']==1
    assert u.returncode==0 and not ledger.exists()


def test_dead_publisher_stops_producer_without_claiming_completion(tmp_path):
    spec,p,u,events,factory,stop,done,ledger=fixture(tmp_path,[None]*8,[1])
    with pytest.raises(RuntimeError,match='publisher exited'):
        controller.run_pipeline(spec,1000,process_factory=factory,clock=lambda:0,sleep=lambda _:None,stop=stop)
    assert p.killed and not done.exists()
    assert any(isinstance(e,list) and e[-1]=='continue-test-gsq' for e in events)


def test_upload_drain_timeout_preserves_ledger_if_cleanup_uncertain(tmp_path):
    spec,p,u,events,factory,stop,done,ledger=fixture(tmp_path,[0],[None]*10)
    def fail_stop(*args,**kwargs):raise OSError('Docker unavailable')
    times=iter([0,0,1001])
    with pytest.raises(TimeoutError):
        controller.run_pipeline(spec,1000,process_factory=factory,clock=lambda:next(times),sleep=lambda _:None,stop=fail_stop)
    assert ledger.exists() and u.killed and done.exists()


def test_expired_deadline_launches_no_process(tmp_path):
    spec,p,u,events,factory,stop,done,ledger=fixture(tmp_path,[0],[0])
    with pytest.raises(TimeoutError,match='already expired'):
        controller.run_pipeline(spec,1000,process_factory=factory,clock=lambda:1000,sleep=lambda _:None,stop=stop)
    assert not events and not ledger.exists()


def test_disconnected_cli_does_not_discard_container_ownership(tmp_path):
    spec,p,u,events,factory,stop,done,ledger=fixture(tmp_path,[None]*10,[1])
    def unavailable(argv,**kwargs):
        events.append(argv);return SimpleNamespace(returncode=1)
    with pytest.raises(RuntimeError,match='publisher exited'):
        controller.run_pipeline(spec,1000,process_factory=factory,clock=lambda:0,sleep=lambda _:None,stop=unavailable)
    stopped=[event[-1] for event in events if isinstance(event,list)]
    assert 'continue-test-gsq-publisher' in stopped and 'continue-test-gsq' in stopped
    assert ledger.exists()
