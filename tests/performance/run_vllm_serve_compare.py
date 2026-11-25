"""Start official vLLM servers sequentially; benchmark each with bench serve."""
from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.getenv('VLLM_SERVE_BENCH_OUTPUT',str(ROOT/'tests/artifacts/vllm_bench_serve_20261031')))
PORT=int(os.getenv('VLLM_SERVE_BENCH_PORT','18080'))
MODEL='/storage2/Qwen3-30B-A3B/'
SERVED='qwen3-moe-benchmark'
FULL_SUITE=os.getenv('VLLM_SERVE_BENCH_SUITE','single')=='full'
# name, input tokens, output tokens, concurrency, formal requests
CASES=[('latency',128,128,1,12),('decode',128,512,6,36),
       ('prefill',2048,128,6,36),('long',2048,512,6,36)] if FULL_SUITE else [('single',2048,128,6,60)]


def stop(proc):
    # This process group was created by this runner; it contains only our server.
    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL)
    try: os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError: pass


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    check=socket.socket()
    try: check.bind(('127.0.0.1',PORT))
    finally: check.close()
    common=[sys.executable,str(ROOT/'tests/performance/vllm_cli_with_groupquant.py'),'serve',MODEL,
            '--host','127.0.0.1','--port',str(PORT),'--served-model-name',SERVED,
            '--dtype','bfloat16','--tensor-parallel-size','2','--max-model-len',str(max(c[1]+c[2] for c in CASES)+1),
            '--max-num-batched-tokens','12288','--max-num-seqs','6',
            '--gpu-memory-utilization','0.70','--no-enable-prefix-caching',
            '--no-enable-chunked-prefill','--seed','0']
    modes=os.getenv('VLLM_SERVE_BENCH_MODES','bf16 current scatter').split()
    for mode in modes:
        if mode not in ('bf16','current','scatter'): raise ValueError(mode)
        env=os.environ.copy()
        # Local HTTP benchmark traffic must not traverse the shell's proxy.
        for key in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
            env.pop(key,None)
        env['NO_PROXY']=env['no_proxy']='127.0.0.1,localhost,::1'
        env.update(ASCEND_RT_VISIBLE_DEVICES=env.get('ASCEND_RT_VISIBLE_DEVICES','4,5'),
                   GROUP_QUANT_CLI_REGISTER='0' if mode=='bf16' else '1',
                   GROUP_QUANT_DISPATCH_SCATTER='1' if mode=='scatter' else '0',
                   GROUP_QUANT_GMM2_GROUP_SIZE='0',GROUP_QUANT_GMM1_BACKEND='auto')
        for key in ('VLLM_TORCH_PROFILER_DIR','GROUP_QUANT_PROFILE_PIPE_UTILIZATION','GROUP_QUANT_BENCH_NATIVE_GMM2'):
            env.pop(key,None)
        command=common.copy()
        if mode!='bf16':
            config=dict(quant_method='group_quant',weight_bits=8,act_bits=8,residual_bits=4,
                        residual_group=2048,quantize_weights=True,quantize_activations=True,
                        quantize_moe=True,quantize_linear=False)
            command+=['--quantization','group_quant','--hf-overrides',json.dumps({'quantization_config':config})]
        (OUT/f'{mode}_server_command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (OUT/f'{mode}_server.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('SERVER_START',mode,proc.pid,flush=True)
            try:
                deadline=time.monotonic()+600
                while True:
                    if proc.poll() is not None: raise RuntimeError(f'{mode} server exited: inspect {log.name}')
                    try:
                        direct=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                        with direct.open(f'http://127.0.0.1:{PORT}/health',timeout=2) as response:
                            if response.status==200: break
                    except Exception:
                        if time.monotonic()>deadline: raise TimeoutError(f'{mode} server readiness timeout')
                        time.sleep(2)
                print('SERVER_READY',mode,flush=True)
                repeats=2 if FULL_SUITE else 1
                for repeat in range(repeats):
                  # Reverse the case order in the second pass to reduce case-order bias.
                  for case,inputs,outputs,concurrency,requests in (CASES if repeat==0 else list(reversed(CASES))):
                    name=f'{mode}_{case}_r{repeat+1}' if FULL_SUITE else mode
                    bench=['vllm','bench','serve','--backend','openai','--host','127.0.0.1','--port',str(PORT),
                       '--model',SERVED,'--tokenizer',MODEL,'--dataset-name','random',
                       '--random-input-len',str(inputs),'--random-output-len',str(outputs),'--random-range-ratio','0',
                       '--num-prompts',str(requests),'--num-warmups',str(concurrency),'--max-concurrency',str(concurrency),
                       '--request-rate','inf','--seed','42','--temperature','0','--ignore-eos',
                       '--percentile-metrics','ttft,tpot,itl,e2el','--metric-percentiles','50,95,99',
                       '--save-result','--save-detailed','--result-dir',str(OUT),
                       '--result-filename',f'{name}.json','--disable-tqdm']
                    (OUT/f'{name}_bench_command.json').write_text(json.dumps(bench,indent=2)+'\n')
                    print('BENCH_START',name,flush=True)
                    with (OUT/f'{name}_bench.log').open('w') as bench_log:
                        subprocess.run(bench,cwd=ROOT,env=env,stdout=bench_log,stderr=subprocess.STDOUT,check=True,timeout=600)
                    result=json.loads((OUT/f'{name}.json').read_text())
                    assert result['completed']==requests and result['failed']==0, name
                    assert result['total_input_tokens']==requests*inputs, name
                    assert result['total_output_tokens']==requests*outputs, name
                    print('BENCH_COMPLETE',name,result['output_throughput'],flush=True)
            finally: stop(proc)
        # Wait for our server's children to release device memory and the port.
        time.sleep(4)

if __name__=='__main__':main()
