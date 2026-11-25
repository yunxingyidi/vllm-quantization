"""Analyze actual activations and actual CANN dynamic-quant outputs; no synthetic data."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'tests/artifacts/activation_outliers_20261120'
OUT=ROOT/'docs/performance_analysis/activation_outliers_20261120'
OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads((SRC/'manifest.json').read_text());files=sorted((SRC/'tensors').glob('*.pt'));assert len(files)==56,len(files)
ratios=[];zero_rates=[];errors=[];rows=[];total_e=total_power=0;total_nonzero=total_lost=0
for path in files:
 d=torch.load(path,map_location='cpu',weights_only=True);x=d['x'].float().numpy();q=d['q'].numpy();sc=d['scale'].float().numpy().reshape(-1,1)
 assert x.shape==(512,2048) and q.shape==x.shape and sc.shape==(512,1)
 assert np.isfinite(x).all() and np.isfinite(sc).all() and (sc>0).all() and q.dtype==np.int8
 a=np.abs(x); med=np.median(a,axis=1);ratio=a.max(axis=1)/np.maximum(med,1e-30)
 recon=q.astype(np.float32)*sc;err=(recon-x)**2;power=(x*x).sum(axis=1)
 nz=x!=0;lost=(q==0)&nz
 zr=lost.sum(axis=1)/nz.sum(axis=1)
 nmse=err.sum(axis=1)/np.maximum(power,1e-30)
 ratios.extend(ratio);zero_rates.extend(zr);errors.extend(nmse)
 total_e+=float(err.sum(dtype=np.float64));total_power+=float((x*x).sum(dtype=np.float64));total_nonzero+=int(nz.sum());total_lost+=int(lost.sum())
 scale_check=float(np.max(np.abs(sc[:,0]-a.max(axis=1)/127)))
 rows.append({'layer':d['layer'],'window':d['window'],'ratio_median':float(np.median(ratio)),'ratio_p95':float(np.percentile(ratio,95)),'nonzero_to_zero_percent':float(100*lost.sum()/nz.sum()),'nmse':float(err.sum(dtype=np.float64)/(x*x).sum(dtype=np.float64)),'scale_vs_absmax_div127_max_absolute_diff':scale_check,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 if d['layer']==24 and d['window']==0:
  example=a.copy();example_x=x.copy();example_recon=recon.copy();example_ratio=ratio.copy()
r=np.array(ratios);z=np.array(zero_rates);edges=np.percentile(r,[0,25,50,75,100]);bins=[]
for i in range(4):
 mask=(r>=edges[i])&((r<=edges[i+1]) if i==3 else (r<edges[i+1]))
 bins.append({'ratio_range':[float(edges[i]),float(edges[i+1])],'tokens':int(mask.sum()),'mean_nonzero_to_zero_percent':float(100*z[mask].mean())})
summary={'manifest':manifest,'samples':len(files),'activation_rows':len(r),'elements':len(r)*2048,'ratio_percentiles':{str(p):float(np.percentile(r,p)) for p in [50,90,95,99,100]},'global_nonzero_to_zero_percent':100*total_lost/total_nonzero,'nmse':total_e/total_power,'sqnr_db':float(10*np.log10(total_power/total_e)),'ratio_quartile_bins':bins,'records':rows,'limitations':['7 preselected layers; first 8 WikiText2 512-token windows; prefill only','BF16 model activations; native dynamic INT8 activation quantization probe, not full W8A8 model output','zeros and reconstruction errors do not independently prove PPL or downstream quality degradation','max/median ratio is a diagnostic statistic, not a definition of erroneous/noisy activation']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
font_manager.fontManager.addfont('/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf')
plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
fig,axs=plt.subplots(1,3,figsize=(15,4.5))
axs[0].plot(example.max(axis=0),color='#C43B35',linewidth=.9,label='跨token最大绝对值');axs[0].plot(np.median(example,axis=0),color='#2879B5',linewidth=.8,label='跨token绝对值中位数');axs[0].set_title('（a）真实激活：通道幅值存在差异',color='#005B96');axs[0].set_xlabel('通道索引（第24层，第0窗口）');axs[0].set_ylabel('激活绝对值');axs[0].legend(frameon=False,fontsize=9);axs[0].grid(alpha=.15)
axs[1].hist(r,bins=np.geomspace(max(r.min(),1),r.max()*1.0001,40),color='#2879B5');axs[1].set_xscale('log');axs[1].set_xlabel('每token最大绝对值 / 绝对值中位数');axs[1].set_ylabel('Token数');axs[1].set_title('（b）大值与普通值的幅度差异',color='#005B96');axs[1].axvline(np.median(r),color='#C43B35',ls='--');axs[1].text(.97,.95,f'中位数 {np.median(r):.1f}倍\nP95 {np.percentile(r,95):.1f}倍',transform=axs[1].transAxes,ha='right',va='top')
vals=[b['mean_nonzero_to_zero_percent'] for b in bins];bars=axs[2].bar(['Q1','Q2','Q3','Q4'],vals,color=['#84B3D3','#5798C4','#2879B5','#C43B35']);axs[2].bar_label(bars,fmt='%.1f%%',padding=5);axs[2].set_ylim(0,max(vals)*1.25);axs[2].set_xlabel('按最大值/中位数比值分位分组（低→高）');axs[2].set_ylabel('原非零激活量化为零的平均比例（%）');axs[2].set_title('（c）原生INT8量化的分辨率损失',color='#005B96')
fig.tight_layout();fig.subplots_adjust(bottom=.20,wspace=.40)
fig.text(.5,.065,'实测：Qwen3-30B-A3B BF16，WikiText2前8×512 tokens，7个预先指定MoE层；NPU1，原生npu_dynamic_quant',ha='center',fontsize=10,color='#555555');fig.text(.5,.02,'激活异常值及局部量化误差诊断；非整网W8A8质量评测，量化为零不等于任务预测错误',ha='center',fontsize=10,color='#555555')
for ext in ['png','svg']:fig.savefig(OUT/f'activation_outliers_measured.{ext}',dpi=220,facecolor='white')
plt.close(fig)
