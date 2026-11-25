"""Validate official detailed serving JSONs and summarize independent repetitions."""
from pathlib import Path
import csv
import json
import math
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'tests/artifacts/vllm_full_suite_20261031'
DEST=ROOT/'docs/performance_analysis/vllm_full_suite'
CASES={'latency':(128,128,1,12),'decode':(128,512,6,36),'prefill':(2048,128,6,36),'long':(2048,512,6,36)}
MODES={'bf16':'BF16','current':'原单组量化','scatter':'优化 dispatch 单组量化'}
METRICS=['duration','output_throughput','mean_ttft_ms','p95_ttft_ms','mean_tpot_ms','p95_tpot_ms','mean_e2el_ms','p95_e2el_ms']
rows=[]
summary={}
for case,(inputs,outputs,concurrency,requests) in CASES.items():
    summary[case]={}
    for mode in MODES:
        runs=[]
        for rep in (1,2):
            path=SOURCE/f'{mode}_{case}_r{rep}.json'
            d=json.loads(path.read_text())
            assert d['completed']==requests and d['failed']==0,path
            assert d['total_input_tokens']==requests*inputs,path
            assert d['total_output_tokens']==requests*outputs,path
            assert d['input_lens']==[inputs]*requests,path
            assert d['output_lens']==[outputs]*requests,path
            assert len(d['generated_texts'])==len(d['ttfts'])==len(d['itls'])==requests,path
            assert not any(d['errors']),path
            assert math.isclose(d['duration']*d['output_throughput'],requests*outputs,rel_tol=1e-7),path
            assert math.isclose(d['mean_e2el_ms'],d['mean_ttft_ms']+(outputs-1)*d['mean_tpot_ms'],rel_tol=1e-6),path
            row=dict(case=case,mode=mode,repeat=rep,input_tokens=inputs,output_tokens=outputs,concurrency=concurrency,completed=requests,failed=0)
            row.update({k:d[k] for k in METRICS})
            rows.append(row)
            runs.append(row)
        s={k:statistics.mean(r[k] for r in runs) for k in METRICS}
        s['output_throughput_runs']=[r['output_throughput'] for r in runs]
        s['output_throughput_range']=[min(s['output_throughput_runs']),max(s['output_throughput_runs'])]
        summary[case][mode]=s
    for mode in ('current','scatter'):
        summary[case][mode]['throughput_vs_bf16_pct']=100*(summary[case][mode]['output_throughput']/summary[case]['bf16']['output_throughput']-1)
    summary[case]['scatter']['throughput_vs_current_pct']=100*(summary[case]['scatter']['output_throughput']/summary[case]['current']['output_throughput']-1)
DEST.mkdir(parents=True,exist_ok=True)
with (DEST/'timings.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(DEST/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
lines=['# 完整 vLLM HTTP 推理负载矩阵','',
'使用安装的官方 `vllm serve` 和未修改的 `vllm bench serve`，覆盖请求提交、调度、prefill、逐 token decode 和 HTTP 流式返回。每个请求生成固定数量 token 后结束；不是单独算子或抽取部分 engine step 的计时。', '',
'模型 Qwen3-30B-A3B，Ascend 910B3 卡 4/5、TP2、BF16 基础 dtype。对比 BF16、原单组 W8A8 GroupQuant MoE、优化 dispatch 单组 W8A8 GroupQuant MoE；只量化 MoE，其余线性层 BF16。GMM1 分组大小 2048（完整 K），GMM2 为本地完整 K384。', '',
'## 吞吐与延迟','',
'下表为两次独立正式压测指标的算术平均；吞吐范围为两次运行的最小值到最大值，不是统计置信区间。p95 在原始 CSV 中逐轮保留，未把两轮 p95 当作合并请求分布的 p95。','',
'| 输入/输出 token | 并发 | 路径 | 输出 token/s（两轮范围） | 平均 TTFT ms | 平均 TPOT ms | 平均请求耗时 s |',
'|---|---:|---|---:|---:|---:|---:|']
for case,(inputs,outputs,concurrency,_) in CASES.items():
    for mode,label in MODES.items():
        s=summary[case][mode];lo,hi=s['output_throughput_range']
        lines.append(f"| {inputs}/{outputs} | {concurrency} | {label} | {s['output_throughput']:.2f} ({lo:.2f}–{hi:.2f}) | {s['mean_ttft_ms']:.2f} | {s['mean_tpot_ms']:.2f} | {s['mean_e2el_ms']/1000:.3f} |")
lines+=['','| 场景 | 原量化 / BF16 吞吐变化 | 优化 / BF16 吞吐变化 | 优化 / 原量化 吞吐变化 |','|---|---:|---:|---:|']
for case,(inputs,outputs,concurrency,_) in CASES.items():
    s=summary[case]
    lines.append(f"| 输入{inputs} / 输出{outputs} / 并发{concurrency} | {s['current']['throughput_vs_bf16_pct']:+.2f}% | {s['scatter']['throughput_vs_bf16_pct']:+.2f}% | {s['scatter']['throughput_vs_current_pct']:+.2f}% |")
lines+=['','## 结果解释','',
'本次完整服务测试没有体现单组量化的明显吞吐优势。原量化各场景与 BF16 的均值差距约在1.3%以内；启用新 dispatch 的配置在两个长输入场景反而更慢。局部算子的加速不能直接等同于服务吞吐加速。', '',
'长输入短输出场景中，优化配置的平均 TTFT 比原量化略低，但 TPOT 更高，完整请求耗时更长；这是整轮测量的现象，并不是新 dispatch 直接导致 decode 变慢的因果证据。短输入场景未触发新 kernel，也存在差异，且所有路径顺序测试、同机存在其他负载。下一步应在真实服务负载下采集实际 dispatch 形状、调用频率和 CPU/NPU 时间线，再定位提交、同步或混合调度开销。', '',
'## 计数与配置核验','',
'共 3 条路径 × 4 个场景 × 2 次正式压测，共 24 份结果、720 个正式请求。全部请求成功、失败为零。逐份检查输入/输出 token 数、逐请求长度、结果数组、错误列表、吞吐乘以时长等于输出总数，以及 TTFT/TPOT/完整请求时间的恒等关系。', '',
'每条路径复用一个服务，max-model-len2561、max-num-batched-tokens12288、max-num-seqs6、gpu-memory-utilization0.70、图模式、关闭 prefix caching 和 chunked prefill。客户端 random 数据集、seed42、固定长度、request-rate=inf、temperature0、ignore-eos。每次正式测试前使用与该场景并发数相同的预热请求；官方还会执行一次初始可用性测试。数据集准备、预热、模型加载不计入正式吞吐。', '',
'第一轮场景顺序为 latency、decode、prefill、long，第二轮反向。路径依次为 BF16、原量化、优化量化，未随机交错重启路径；两轮不足以证明稳定的小幅加速。同机卡6/7存在其他任务，主机负载可能影响结果。延迟同时包含服务调度与 HTTP 处理，不等于设备 kernel 时长。本次不新增准确率/PPL测量。', '',
'优化 dispatch 仅在 legacy AllGather、原 token 行数至少2048等条件满足时启用。128 token 短输入和小 batch decode 回退原 dispatch；它们作为整体运行波动及量化成本的对照，不能把短输入场景的差异归因于新的 scatter kernel。长输入场景可能使用优化路径，但本轮未采集设备 trace 来分离各阶段的因果贡献。', '',
'## 复现与产物','',
'```bash',
'VLLM_SERVE_BENCH_SUITE=full VLLM_SERVE_BENCH_OUTPUT=tests/artifacts/vllm_full_suite_20261031 bash tests/performance/run_vllm_serve_compare.sh',
'python tests/performance/summarize_vllm_full_suite.py',
'```','',
'原始官方 JSON、逐请求详情、服务日志、客户端日志和完整命令位于 `tests/artifacts/vllm_full_suite_20261031/`。本目录包含 `timings.csv` 和 `summary.json`，保留所有逐轮指标。仅停止本 runner 启动的服务进程组。']
(DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2,ensure_ascii=False))
