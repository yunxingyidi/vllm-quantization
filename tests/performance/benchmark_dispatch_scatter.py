"""Validate and time native routing versus metadata-route + quantize/scatter.

No production dispatcher change. Each timed pipeline constructs its own mapping
and expert counts. Shape: finite BF16 [M,H], full expert range, top-k selections.
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_npu
from tests.performance.benchmark_dispatch_quant import make_input, router


def load():
    path = Path(__file__).resolve().parents[2] / 'custom_op/dispatch_quant_scatter/build/libdispatch_quant_scatter_ops.so'
    torch.ops.load_library(str(path))


def candidate(x, ids, experts):
    _, mapping, counts, _ = router(x[:, :1].contiguous(), ids, experts)
    q, scale = torch.ops.group_quant.dispatch_quant_scatter(x, mapping, ids.shape[1])
    return q, mapping, counts, scale


def check(m, h, topk, experts, mode):
    x, ids = make_input(m, h, topk, experts, mode, 42)
    ref = router(x, ids, experts, quant_mode=1)
    actual = candidate(x, ids, experts)
    torch.npu.synchronize()
    fq, fm, fc, fs = [t.cpu() for t in ref]
    aq, am, ac, ass = [t.cpu() for t in actual]
    order = torch.argsort(ids.cpu().reshape(-1), stable=True)
    source = order // topk
    expected_counts = torch.bincount(ids.cpu().reshape(-1).long(), minlength=experts)
    assert torch.equal(fm, am) and torch.equal(fc, ac)
    assert torch.equal(ac.long(), expected_counts)
    assert torch.equal(am.long()[order], torch.arange(m*topk))
    # Separate native quant checks numerical semantics and exact top-k reuse.
    nq, ns = torch_npu.npu_dynamic_quant(x)
    nq, ns = nq.cpu()[source], ns.cpu().reshape(-1)[source]
    diff = (aq.short()-fq.short()).abs()
    assert torch.isfinite(ass).all()
    assert torch.allclose(ass, fs, rtol=1e-6, atol=1e-8), (ass[:16],fs[:16])
    assert diff.max() <= 1, int(diff.max())
    assert (aq.short()-nq.short()).abs().max() <= 1
    # Decode the expert-major output into original (token, top-k) order.
    restored_q = aq[am.long()].reshape(m,topk,h)
    restored_s = ass[am.long()].reshape(m,topk)
    assert torch.equal(restored_q,restored_q[:,:1].expand_as(restored_q))
    assert torch.equal(restored_s,restored_s[:,:1].expand_as(restored_s))
    return dict(m=m,h=h,topk=topk,routing=mode,max_lsb=int(diff.max()),
                mismatch_elements=int((diff!=0).sum()),total_elements=aq.numel(),
                scale_max_abs=float((ass-fs).abs().max()),
                mapping_and_counts='exact',topk_quant_and_scale_reuse='exact',
                native_scale_max_abs=float((ass-ns).abs().max()))


def capture(fn):
    for _ in range(5): out=fn()
    torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph): out=fn()
    return graph,out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m',type=int,nargs='+',default=[6,2048,10240])
    p.add_argument('--iters',type=int,default=50)
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--require-exact',action='store_true',
                   help='Require native-routing bitwise quant/scale agreement')
    a=p.parse_args()
    load()
    checks=[]
    for m,h,k,mode in [(1,64,1,'hotspot'),(6,2048,8,'uniform'),(17,2048,2,'uniform'),
                        (65,2048,8,'hotspot'),(257,2048,8,'uniform')]:
        checks.append(check(m,h,k,128,mode))
        if a.require_exact:
            assert checks[-1]['max_lsb']==0 and checks[-1]['scale_max_abs']==0
        print('CHECK',checks[-1],flush=True)
    # Empty tensor launch must return correctly shaped empty outputs.
    z=torch.empty((0,2048),dtype=torch.bfloat16,device='npu')
    zm=torch.empty((0,),dtype=torch.int32,device='npu')
    zq,zs=torch.ops.group_quant.dispatch_quant_scatter(z,zm,8)
    assert zq.shape==(0,2048) and zs.shape==(0,)
    # Replay with changed values and changed expert assignments. No mapping,
    # quantized output, or scale may survive from the captured input.
    gx,gi=make_input(17,2048,8,128,'uniform',3)
    replay,replay_out=capture(lambda:candidate(gx,gi,128))
    newx,newi=make_input(17,2048,8,128,'hotspot',9)
    gx.copy_(newx); gi.copy_(newi)
    replay.replay()
    expected=router(gx,gi,128,quant_mode=1)
    torch.npu.synchronize()
    assert torch.equal(replay_out[1].cpu(),expected[1].cpu())
    assert torch.equal(replay_out[2].cpu(),expected[2].cpu())
    assert (replay_out[0].cpu().short()-expected[0].cpu().short()).abs().max()<=1
    assert torch.allclose(replay_out[3].cpu(),expected[3].cpu(),rtol=1e-6,atol=1e-8)
    print('CHANGED_INPUT_GRAPH passed',flush=True)
    results=[]
    for m in a.m:
        x,ids=make_input(m,2048,8,128,'uniform',42)
        funcs={'current':lambda:router(x,ids,128,quant_mode=1),
               'quantize_scatter':lambda:candidate(x,ids,128)}
        graphs={name:capture(fn) for name,fn in funcs.items()}
        samples={name:[] for name in funcs}
        for repeat in range(a.repeats):
            for name in list(funcs)[::1 if repeat%2==0 else -1]:
                graph,out=graphs[name]
                torch.npu.synchronize()
                start=time.perf_counter()
                for _ in range(a.iters): graph.replay()
                torch.npu.synchronize()
                samples[name].append((time.perf_counter()-start)*1000/a.iters)
        # Validate captured outputs again after repeated graph replay.
        oq,om,oc,os=graphs['current'][1]
        nq,nm,nc,ns=graphs['quantize_scatter'][1]
        assert torch.equal(om.cpu(),nm.cpu()) and torch.equal(oc.cpu(),nc.cpu())
        assert (oq.cpu().short()-nq.cpu().short()).abs().max()<=1
        assert torch.allclose(os.cpu(),ns.cpu(),rtol=1e-6,atol=1e-8)
        row={'m':m,'h':2048,'topk':8,'experts':128,'iters':a.iters,
             'repeats':a.repeats,'graph':True,'samples_ms':samples,
             'quant_bitwise_equal':torch.equal(oq.cpu(),nq.cpu()),
             'scale_bitwise_equal':torch.equal(os.cpu(),ns.cpu()),
             'median_ms':{k:statistics.median(v) for k,v in samples.items()}}
        if a.require_exact:
            assert row['quant_bitwise_equal'] and row['scale_bitwise_equal']
        results.append(row)
        print('RESULT',row,flush=True)
    a.result.parent.mkdir(parents=True,exist_ok=True)
    a.result.write_text(json.dumps({'checks':checks,'results':results,
        'timing':'complete pipelines, ACL Graph replay wall time; alternating order'},indent=2)+'\n')

if __name__=='__main__': main()
