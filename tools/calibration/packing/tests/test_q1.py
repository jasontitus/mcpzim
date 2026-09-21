import ctypes
import hashlib
import os
from pathlib import Path
import struct
import numpy as np
import pytest
from packing.q1 import pack, unpack, from_sign_logits, reorder_qwen_candidate, v_head_permutation
from packing.gguf import write, string

ROOT = Path(__file__).resolve().parents[4]
LIB = ROOT / 'ios/LocalPackages/llama.cpp-swift/llama.xcframework/macos-arm64_x86_64/llama.framework/llama'


def test_sign_order_zero_logits_negative_and_exact_scales():
    logits = np.zeros((2,256), dtype=np.float32)
    logits[0, [0,7,8,127,128,255]] = 1
    scales = np.array([[.5, -.25], [1, 0]], dtype=np.float32)
    packed, metrics = from_sign_logits(logits, scales)
    assert packed.shape == (2,36) and metrics['bytes'] == 72
    assert packed[0,2] == 129 and packed[0,3] == 1 and packed[0,17] == 128
    expected = np.where(logits > 0,1.,-1.) * np.repeat(scales,128,axis=1)
    np.testing.assert_array_equal(unpack(packed), expected)
    assert metrics['changed_groups'] == 0


def test_export_scale_loss_requires_explicit_acceptance():
    bits = np.zeros((1,16),np.uint8)
    with pytest.raises(ValueError,match='explicit acceptance'):
        pack(bits, [[1.0001]])
    data, metrics = pack(bits, [[1.0001]], allow_scale_loss=True)
    assert metrics['changed_groups'] == 1 and metrics['sum_squared_weight_error'] > 0
    with pytest.raises(ValueError,match='overflows'):
        pack(bits, [[1e8]], allow_scale_loss=True)
    with pytest.raises(ValueError,match='explicit acceptance'):
        pack(bits, [[1e-12]])


def test_shipped_runtime_dequantizer_matches_numpy():
    if not LIB.exists(): pytest.skip('Shipped macOS framework unavailable')
    runtime = ctypes.CDLL(str(LIB))
    fn = runtime.dequantize_row_q1_0
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn.restype = None
    rng = np.random.default_rng(90)
    bits = rng.integers(0,256,(7,64),dtype=np.uint8)
    scales = rng.uniform(-2,2,(7,4)).astype(np.float16).astype(np.float32)
    data,_ = pack(bits,scales)
    result = np.empty((7,512), np.float32)
    fn(data.ctypes.data,result.ctypes.data,result.size)
    np.testing.assert_array_equal(result,unpack(data))


def test_qwen_vhead_permutation_preserves_learned_groups():
    rng = np.random.default_rng(5)
    bits = rng.integers(0,256,(128,64),dtype=np.uint8)
    scales = np.arange(128*4,dtype=np.float32).reshape(128,4)/16
    a,_=pack(bits,scales)
    b,s=reorder_qwen_candidate('model.language_model.layers.0.linear_attn.out_proj.weight',bits,scales,k_heads=2,v_heads=4)
    c,_=pack(b,s)
    np.testing.assert_array_equal(unpack(c),unpack(a)[:,v_head_permutation(2,4,128)])
    with pytest.raises(ValueError,match='splits learned'):
        reorder_qwen_candidate('x.linear_attn.out_proj.weight',bits,scales,k_heads=2,v_heads=4,head_v_dim=64)


def test_actual_gsq_hard_candidate_roundtrip():
    path=Path(os.environ.get('ZIMFO_GSQ_SOURCE','/tmp/zimfo-gsq-research'))/'src/quantization/gumbel_quantizer_1bit.py'
    if not path.exists(): pytest.skip('Pinned GSQ checkout unavailable')
    import importlib.util, torch
    spec=importlib.util.spec_from_file_location('gsq_onebit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(41)
    q=torch.randn(11,256,dtype=torch.bfloat16)
    scales=torch.linspace(.125,1.25,22).reshape(11,2)
    quantizer=module.GumbelQuantizer1Bit(q,scales,128,.3,1.,'cpu',torch.bfloat16)
    hard,effective=quantizer.get_hard_weights()
    packed,metrics=from_sign_logits(quantizer.sign_logits.detach().float().numpy(),effective.detach().float().numpy())
    np.testing.assert_array_equal(unpack(packed),hard.detach().float().numpy())
    assert metrics['sum_squared_weight_error'] == 0


def test_gguf_serialized_size_and_failed_hash_never_publish(tmp_path):
    payload=tmp_path/'weight.bin'; payload.write_bytes(bytes(18))
    tensor={'name':'weight','shape':[1,128],'type':41,'payload':str(payload),'sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}
    metadata=string('general.architecture')+struct.pack('<I',8)+string('llama')
    path=tmp_path/'test.gguf'
    report=write(path,metadata,1,[tensor])
    assert path.stat().st_size == report['bytes'] and report['tensor_payload_bytes']==18
    with pytest.raises(FileExistsError): write(path,metadata,1,[tensor])
    with pytest.raises(ValueError,match='hash'):
        write(tmp_path/'bad.gguf',metadata,1,[{**tensor,'sha256':'0'*64}])
    assert not (tmp_path/'bad.gguf').exists() and not (tmp_path/'bad.gguf.partial').exists()


def test_preexisting_partial_is_not_owned_or_deleted(tmp_path):
    payload=tmp_path/'payload';payload.write_bytes(bytes(18))
    entries=[{'name':'w','shape':[1,128],'type':41,'payload':str(payload),'sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}]
    partial=tmp_path/'out.gguf.partial';partial.write_bytes(b'another writer')
    with pytest.raises(FileExistsError): write(tmp_path/'out.gguf',b'',0,entries)
    assert partial.read_bytes()==b'another writer'


def test_metadata_alignment_count_and_serialized_budget(tmp_path):
    from packing.gguf import validate_metadata,serialized_size
    metadata=string('general.alignment')+struct.pack('<II',4,64)
    with pytest.raises(ValueError,match='alignment'): validate_metadata(metadata,1)
    with pytest.raises(ValueError,match='trailing'): validate_metadata(metadata,0)
    with pytest.raises(ValueError,match='Truncated'): validate_metadata(metadata[:-1],1)
    t={'name':'weight','shape':[3,256],'type':41}
    assert serialized_size(b'',0,[t])==224
    payload=tmp_path/'matrix.bin';payload.write_bytes(bytes(108))
    entry={**t,'payload':str(payload),'sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}
    assert write(tmp_path/'sized.gguf',b'',0,[entry])['bytes']==224


def test_qwen_candidate_layout_matches_pinned_converter():
    source=Path(os.environ.get('ZIMFO_PRISM_SOURCE','/tmp/zimfo-prism-packing'))
    if not source.exists(): pytest.skip('Pinned converter checkout unavailable')
    import torch
    from packing.export_qwen import load_converter
    cls,g,_=load_converter(source)
    converter=object.__new__(cls)
    converter.hparams={'linear_num_key_heads':2,'linear_num_value_heads':4,
        'linear_key_head_dim':128,'linear_value_head_dim':128}
    converter.tensor_map=g.get_tensor_name_map(g.MODEL_ARCH.QWEN35,1)
    converter.fuse_gate_up_exps=False
    rng=np.random.default_rng(33)
    for suffix,shape in [('in_proj_qkv',(1024,128)),('in_proj_z',(512,128)),
                         ('in_proj_a',(4,128)),('in_proj_b',(4,128)),('out_proj',(128,512))]:
        name=f'model.layers.0.linear_attn.{suffix}.weight'
        codes=rng.integers(0,256,(shape[0],shape[1]//8),dtype=np.uint8)
        scales=rng.uniform(.01,.1,(shape[0],shape[1]//128)).astype(np.float16).astype(np.float32)
        original,_=pack(codes,scales)
        expected=list(converter.modify_tensors(torch.from_numpy(unpack(original)),name,0))
        bits,reordered=reorder_qwen_candidate(name,codes,scales,k_heads=2,v_heads=4)
        actual,_=pack(bits,reordered)
        assert len(expected)==1
        np.testing.assert_array_equal(unpack(actual),expected[0][1].numpy())
    norm=torch.tensor([.03125,-.125],dtype=torch.float32)
    got=list(converter.modify_tensors(norm,'model.norm.weight',None))
    np.testing.assert_array_equal(got[0][1].numpy(),norm.numpy()+1)
    a=torch.tensor([0.,1.,2.,3.])
    got=list(converter.modify_tensors(a,'model.layers.0.linear_attn.A_log',0))
    np.testing.assert_allclose(got[0][1].numpy(),-np.exp(a.numpy()[[0,2,1,3]]),rtol=1e-6)


def test_solver_safetensors_candidate_contract_and_hash(tmp_path):
    import torch
    from safetensors.torch import save_file
    from packing.export_qwen import load_candidate
    path=tmp_path/'candidate.safetensors'
    save_file({'codes':torch.full((3,16),129,dtype=torch.uint8),
               'scales':torch.full((3,1),.125,dtype=torch.bfloat16)},str(path),
        metadata={'format':'zimfo-q1-v1','group_size':'128','bit_order':'little','effective_scale_dtype':'BF16'})
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    codes,scales=load_candidate(path,[3,128],digest)
    packed,error=pack(codes,scales)
    assert packed.shape==(3,18) and error['changed_groups']==0
    with pytest.raises(ValueError,match='hash'):load_candidate(path,[3,128],'0'*64)
    with pytest.raises(ValueError,match='dimensions'):load_candidate(path,[4,128],digest)


def test_rco_cost_manifest_is_exact_for_mixed_q1_bf16_choices():
    source=Path(os.environ.get('ZIMFO_PRISM_SOURCE','/tmp/zimfo-prism-packing'))
    if not source.exists(): pytest.skip('Pinned converter checkout unavailable')
    from packing.export_qwen import load_converter,cost_manifest
    from packing.gguf import serialized_size
    cls,g,_=load_converter(source)
    converter=object.__new__(cls);converter.hparams={'linear_num_key_heads':2,'linear_num_value_heads':4,'linear_key_head_dim':128,'linear_value_head_dim':128}
    converter.tensor_map=g.get_tensor_name_map(g.MODEL_ARCH.QWEN35,1);converter.fuse_gate_up_exps=False
    originals={'model.language_model.embed_tokens.weight':{'shape':[32,128]},'lm_head.weight':{'shape':[32,128]},'model.language_model.norm.weight':{'shape':[128]}}
    costs=cost_manifest(converter,b'',0,originals)
    entries=[{'name':'token_embd.weight','shape':[32,128],'type':41},
             {'name':'output.weight','shape':[32,128],'type':30},
             {'name':'output_norm.weight','shape':[128],'type':0}]
    predicted=costs['fixed_container_bytes']+costs['tensors']['model.language_model.embed_tokens.weight']['q1_bytes']+costs['tensors']['lm_head.weight']['bf16_bytes']
    assert predicted==serialized_size(b'',0,entries)


def test_archived_prism_receipt_checks_files_revision_and_extra_code(tmp_path):
    import json
    from packing.export_qwen import PRISM,verify_prism_source
    files={}
    for relative in ('conversion/base.py','conversion/qwen.py','conversion/__init__.py','gguf-py/gguf/constants.py','gguf-py/gguf/__init__.py','LICENSE'):
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'# fixture')
        files[relative]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt=tmp_path/'zimfo-prism-source.json';receipt.write_text(json.dumps({'revision':PRISM,'files':files}))
    assert verify_prism_source(tmp_path)==PRISM
    (tmp_path/'injected.py').write_bytes(b'pass')
    with pytest.raises(ValueError,match='Unverified Python'):verify_prism_source(tmp_path)
    (tmp_path/'injected.py').unlink();(tmp_path/'conversion/qwen.py').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):verify_prism_source(tmp_path)


def test_row_streamed_rtn_fixture_matches_declared_algorithm(tmp_path,monkeypatch):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from packing import rtn_fixture
    monkeypatch.setattr(rtn_fixture,'require_space',lambda *a,**kw:None)
    source=tmp_path/'original.safetensors';target=tmp_path/'candidate.safetensors'
    x=torch.arange(3*256,dtype=torch.float32).reshape(3,256).sub(384).bfloat16()
    save_file({'weight':x},str(source));original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    result=rtn_fixture.write_candidate(source,'weight',target,[3,256],rows=2)
    with safe_open(str(target),framework='pt') as f:
        assert f.metadata()['purpose']=='EXPORT_RUNTIME_COMPATIBILITY_ONLY_NOT_GSQ'
        np.testing.assert_array_equal(f.get_tensor('codes').numpy(),np.packbits((x.float()>0).numpy(),axis=-1,bitorder='little'))
        expected=x.float().abs().reshape(3,2,128).mean(-1).bfloat16()
        torch.testing.assert_close(f.get_tensor('scales'),expected,rtol=0,atol=0)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original_hash
    assert result['bytes']==target.stat().st_size
    with pytest.raises(FileExistsError):rtn_fixture.write_candidate(source,'weight',target,[3,256],rows=2)
