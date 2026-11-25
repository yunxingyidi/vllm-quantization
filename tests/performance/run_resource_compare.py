"""Isolated-card service resource experiment; no production package edits.

Use --devices with one or two exclusively available cards. Never evicts other jobs.
Performance runs are unprofiled. Short device timelines are captured separately.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tests.performance.run_vllm_serve_compare import stop


def smi():
    result=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True,timeout=15)
    cards={};card=None;process_section=False
    for line in result.stdout.splitlines():
        if 'Process id' in line:process_section=True
        if process_section:
            m=re.match(r'\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|',line)
            if m and int(m[1]) in cards:cards[int(m[1])]['pids'].append(int(m[2]))
            continue
        m=re.match(r'\|\s*(\d+)\s+910\w+',line)
        if m:card=int(m[1]);cards[card]={'pids':[]}
        if card is not None and re.search(r'[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:',line):
            fields=line.split('|');nums=re.findall(r'\d+',fields[3])
            if len(nums)>=5:cards[card].update(aicore_percent=int(nums[0]),hbm_used_mib=int(nums[-2]),hbm_total_mib=int(nums[-1]))
    return cards,result.stdout


class Monitor:
    def __init__(self,path,devices):
        self.path=path;self.devices=devices;self.phase='startup';self.rows=[];self.done=threading.Event()
    def run(self):
        with self.path.open('w') as f:
            while not self.done.is_set():
                ts=time.time();phase=self.phase
                try:
                    cards,_=smi()
                    for device in self.devices:
                        row=dict(time=ts,phase=phase,device=device,**cards[device]);self.rows.append(row)
                        f.write(json.dumps(row)+'\n');f.flush()
                except Exception as exc:
                    f.write(json.dumps(dict(time=ts,error=str(exc)))+'\n');f.flush()
                self.done.wait(max(.05,1-(time.time()-ts)))
    def start(self):self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def close(self):self.done.set();self.thread.join(timeout=20)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--devices',required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'tests/artifacts/resource_compare_20261118')
    ap.add_argument('--port',type=int,default=18083)
    ap.add_argument('--modes',nargs='+',choices=['bf16','current','scatter'],default=['bf16','current','scatter'])
    ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--preflight-only',action='store_true')
    args=ap.parse_args();devices=[int(x) for x in args.devices.split(',')]
    if len(devices) not in (1,2) or len(set(devices))!=len(devices):ap.error('one or two distinct physical cards are required')
    tp=len(devices);concurrency=1 if tp==1 else 6;requests=6 if tp==1 else 36
    kv_blocks=32 if tp==1 else 256
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    def idle_guard():
        cards,raw=smi();(out/'last_preflight_smi.txt').write_text(raw)
        for i in devices:
            if i not in cards or cards[i].get('pids') or cards[i].get('aicore_percent',100)>0:
                raise RuntimeError(f'Card {i} is not idle/exclusive; no server launched. Status={cards.get(i)}')
        return cards
    baseline=idle_guard()
    if args.preflight_only:print(json.dumps(baseline,indent=2));return
    sock=socket.socket()
    try:sock.bind(('127.0.0.1',args.port))
    finally:sock.close()
    base=f'http://127.0.0.1:{args.port}'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def rpc(label,reset=False):
        req=urllib.request.Request(base+'/collective_rpc',data=json.dumps({'method':'resource_snapshot','args':[label,'1' if reset else '0'],'timeout':60}).encode(),headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=70) as response:return json.load(response)['results']
    manifest={'devices':devices,'baseline':baseline,'model':'/storage2/Qwen3-30B-A3B/','block_size':128,'tp':tp,'concurrency':concurrency,'kv_blocks':kv_blocks,'cases':[['prefill',2048,128,concurrency,requests],['decode',128,512,concurrency,requests]],'repeats':args.repeats,'profile_case':[2048,24,concurrency,concurrency],'cpu_staging_for_quantized_load':os.getenv('GROUP_QUANT_STAGE_BF16_ON_CPU')=='1','notes':['fixed KV capacity across all modes','allocator transient peak includes activations and workspaces','smi counters use actual query timestamps','timeline run separate from throughput runs']}
    manifest_file=out/'manifest.json'
    if manifest_file.exists():
        previous=json.loads(manifest_file.read_text());manifest['previous_launch_manifest']=previous
    manifest_file.write_text(json.dumps(manifest,indent=2))
    for mode in args.modes:
        idle_guard()
        dest=out/mode;dest.mkdir(exist_ok=True)
        env=os.environ.copy()
        for key in ['http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','GROUP_QUANT_BENCH_NATIVE_GMM2']:env.pop(key,None)
        env.update(ASCEND_RT_VISIBLE_DEVICES=args.devices,NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost',VLLM_SERVER_DEV_MODE='1',
                   GROUP_QUANT_CLI_REGISTER='0' if mode=='bf16' else '1',GROUP_QUANT_DISPATCH_SCATTER='1' if mode=='scatter' else '0',
                   GROUP_QUANT_GMM2_GROUP_SIZE='0',GROUP_QUANT_GMM1_BACKEND='auto',GROUP_QUANT_RESOURCE_OUTPUT=str(dest/'memory'),
                   VLLM_TORCH_PROFILER_DIR=str(dest/'profile'),GROUP_QUANT_PROFILE_PIPE_UTILIZATION='1',GROUP_QUANT_SERVING_PROFILE_SHAPES='1',PYTHONUNBUFFERED='1')
        env['PYTHONPATH']=str(ROOT/'tests/performance/profiler_patch')+':'+str(ROOT)+':'+env.get('PYTHONPATH','')
        command=[sys.executable,str(ROOT/'tests/performance/vllm_cli_with_groupquant.py'),'serve',manifest['model'],'--host','127.0.0.1','--port',str(args.port),'--served-model-name','resource-benchmark','--dtype','bfloat16','--tensor-parallel-size',str(tp),'--max-model-len','2177','--max-num-batched-tokens',str(2177 if tp==1 else 12288),'--max-num-seqs',str(concurrency),'--gpu-memory-utilization','0.99' if tp==1 else '0.70','--block-size','128','--num-gpu-blocks-override',str(kv_blocks),'--no-enable-prefix-caching','--no-enable-chunked-prefill','--seed','0','--worker-extension-cls','tests.performance.resource_probe.ResourceWorkerExtension']
        if mode!='bf16':
            config=dict(quant_method='group_quant',weight_bits=8,act_bits=8,residual_bits=4,residual_group=2048,quantize_weights=True,quantize_activations=True,quantize_moe=True,quantize_linear=False)
            command+=['--quantization','group_quant','--hf-overrides',json.dumps({'quantization_config':config})]
        (dest/'server_command.json').write_text(json.dumps(command,indent=2))
        monitor=Monitor(dest/'device_samples.jsonl',devices);monitor.start()
        with (dest/'server.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('START',mode,proc.pid,flush=True)
            try:
                deadline=time.monotonic()+900
                while True:
                    if proc.poll() is not None:raise RuntimeError(f'{mode} server exited: {log.name}')
                    try:
                        with opener.open(base+'/health',timeout=2) as response:
                            if response.status==200:break
                    except Exception:
                        if time.monotonic()>deadline:raise TimeoutError(mode)
                        time.sleep(2)
                print('READY',mode,flush=True);monitor.phase='ready'
                (dest/'ready_memory.json').write_text(json.dumps(rpc('ready'),indent=2))
                def bench(name,inputs,outputs,concurrency,requests,profile=False,warm=False):
                    cmd=['vllm','bench','serve','--backend','openai','--host','127.0.0.1','--port',str(args.port),'--model','resource-benchmark','--tokenizer',manifest['model'],'--dataset-name','random','--random-input-len',str(inputs),'--random-output-len',str(outputs),'--random-range-ratio','0','--num-prompts',str(requests),'--num-warmups','0','--max-concurrency',str(concurrency),'--request-rate','inf','--seed','42','--temperature','0','--ignore-eos','--save-result','--save-detailed','--result-dir',str(dest),'--result-filename',name+'.json','--disable-tqdm']
                    if profile:cmd+=['--profile']
                    (dest/(name+'_command.json')).write_text(json.dumps(cmd,indent=2))
                    monitor.phase=name+'_prepare'
                    memory_start=rpc(name+'_before',True)
                    times={};active=False
                    with (dest/(name+'.log')).open('w') as bf:
                        bp=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                        watchdog=threading.Timer(600,bp.kill)
                        watchdog.start()
                        try:
                            for line in bp.stdout:
                                bf.write(line);bf.flush()
                                if 'Starting main benchmark run' in line:
                                    monitor.phase=name;times['start']=time.time();active=True
                                if active and 'Benchmark duration (s)' in line:
                                    times['end']=time.time();monitor.phase=name+'_finish';active=False
                            code=bp.wait(timeout=600)
                            if code:raise RuntimeError(f'bench failed {name}: {code}')
                        finally:
                            watchdog.cancel()
                            if bp.poll() is None:bp.terminate();bp.wait(timeout=20)
                    memory_end=rpc(name+'_after')
                    (dest/(name+'_memory.json')).write_text(json.dumps({'before':memory_start,'after':memory_end,'monitor_window':times},indent=2))
                    result=json.loads((dest/(name+'.json')).read_text())
                    assert result['completed']==requests and result['failed']==0
                    assert result['input_lens']==[inputs]*requests and result['output_lens']==[outputs]*requests
                    print('COMPLETE',mode,name,result.get('output_throughput'),flush=True)
                for repeat in range(args.repeats):
                    cases=manifest['cases'] if repeat%2==0 else list(reversed(manifest['cases']))
                    for case,ins,outs,conc,requests in cases:
                        bench(f'warm_{case}_{repeat}',ins,outs,conc,conc,warm=True)
                        bench(f'{case}_r{repeat+1}',ins,outs,conc,requests)
                bench('warm_profile',2048,24,concurrency,concurrency,warm=True)
                bench('timeline',2048,24,concurrency,concurrency,profile=True)
            finally:
                monitor.phase='shutdown';stop(proc);monitor.close()
        time.sleep(5)
    print('ALL_COMPLETE',out,flush=True)


if __name__=='__main__':main()
