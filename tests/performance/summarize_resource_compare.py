"""Summarize allocator snapshots, actual-time device samples and unprofiled HTTP results."""
import argparse,csv,json,statistics
from decimal import Decimal
from pathlib import Path

def percentile(values,q):
    vals=sorted(values)
    if not vals:return None
    x=(len(vals)-1)*q;i=int(x);j=min(i+1,len(vals)-1)
    return vals[i]+(vals[j]-vals[i])*(x-i)

def timeline(path):
    rows=list(csv.DictReader(path.open()))
    spans=sorted((Decimal(r['Start Time(us)'].strip()),Decimal(r['Start Time(us)'].strip())+Decimal(r['Duration(us)'])) for r in rows)
    if not spans:return None
    busy=0;left,right=spans[0]
    for a,b in spans[1:]:
        if a>right:busy+=right-left;left,right=a,b
        else:right=max(right,b)
    busy+=right-left;window=max(b for a,b in spans)-spans[0][0]
    return {'kernel_count':len(rows),'first_to_last_kernel_ms':float(window/1000),'any_kernel_busy_ms':float(busy/1000),'no_kernel_percent':float(100*(1-busy/window)),'note':'first-to-last device kernel window, includes communication as busy; no profiling throughput claim'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args();result={}
    for mode in ['bf16','current','scatter']:
        root=args.root/mode
        ready_file=root/'ready_memory.json'
        if not ready_file.exists():continue
        ready=json.loads(ready_file.read_text());samples=[json.loads(l) for l in (root/'device_samples.jsonl').read_text().splitlines()]
        entry={'ready_by_rank':ready,'cases':{},'profiles':{}}
        for memory_file in sorted(root.glob('*_r*_memory.json')):
            name=memory_file.stem.removesuffix('_memory');mem=json.loads(memory_file.read_text());perf=json.loads((root/(name+'.json')).read_text())
            a=mem['monitor_window']['start'];b=mem['monitor_window']['end']
            item={'performance':{k:perf.get(k) for k in ['completed','failed','duration','output_throughput','mean_ttft_ms','p95_ttft_ms','mean_tpot_ms']},'ranks':[],'devices':{},'monitor_window_seconds':b-a}
            known_pids={x['pid'] for x in mem['after']}
            if item['performance']['p95_ttft_ms'] is None:
                item['performance']['p95_ttft_ms']=percentile([x*1000 for x in perf['ttfts']],.95)
            for start,end in zip(mem['before'],mem['after']):
                assert start['rank']==end['rank']
                item['ranks'].append({'rank':end['rank'],'baseline_allocated_bytes':start['allocated'],'peak_allocated_bytes':end['peak_allocated'],'temporary_allocations_peak_increment_bytes':max(0,end['peak_allocated']-start['allocated']),'peak_reserved_bytes':end['peak_reserved'],'note':'increment includes activations and workspace; graph-persistent allocations reside in baseline'})
            for device in sorted({x['device'] for x in samples if 'device'in x}):
                ss=sorted([x for x in samples if x.get('device')==device and x.get('phase')==name and a<=x['time']<=b],key=lambda x:x['time'])
                vals=[x['aicore_percent'] for x in ss]
                weights=[min(b,ss[i+1]['time'] if i+1<len(ss) else b)-x['time'] for i,x in enumerate(ss)]
                coverage=sum(weights)
                item['devices'][device]={'samples':len(ss),'sample_coverage_seconds':coverage,'aicore_mean_percent':sum(v*w for v,w in zip(vals,weights))/coverage if coverage else None,'aicore_p50_percent':percentile(vals,.5),'aicore_p95_percent':percentile(vals,.95),'zero_util_sample_fraction':sum(v==0 for v in vals)/len(vals) if vals else None,'peak_sampled_hbm_mib':max([x['hbm_used_mib'] for x in ss],default=None),'observed_pids':sorted({p for x in ss for p in x['pids']}),'note':'zero-util sample fraction is not trace-derived device idle time; smi polling may miss short memory peaks'}
                item['devices'][device]['unexpected_pids']=sorted({p for x in ss for p in x['pids']}-known_pids)
                item['devices'][device]['aicore_min_percent']=min(vals,default=None)
                item['devices'][device]['aicore_max_percent']=max(vals,default=None)
            entry['cases'][name]=item
        for path in (root/'profile').glob('*/ASCEND_PROFILER_OUTPUT/kernel_details.csv'):
            entry['profiles'][path.parent.parent.name]=timeline(path)
        result[mode]=entry
    (args.root/'resource_summary.json').write_text(json.dumps(result,indent=2))
    print('Summarized',list(result))

if __name__=='__main__':main()
