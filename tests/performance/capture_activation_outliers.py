"""Read-only diagnostic hook: native dynamic INT8 quantization of actual BF16 MoE inputs."""
import os,json,hashlib
from pathlib import Path
LAYERS={0,8,16,24,32,40,47}
class ActivationCaptureExtension:
 def install_activation_capture(self,output):
  import torch,torch_npu
  self._activation_handles=[];self._activation_counts={};out=Path(output);out.mkdir(parents=True,exist_ok=True)
  def hook(name,layer):
   def capture(module,args):
    x=args[0]
    if x.ndim!=2 or x.shape[0]!=512:return
    count=self._activation_counts.get(layer,0)
    if count>=8:raise RuntimeError('unexpected extra prefill')
    q,scale=torch_npu.npu_dynamic_quant(x)
    data={'x':x.detach().cpu(),'q':q.detach().cpu(),'scale':scale.detach().cpu(),'module':name,'layer':layer,'window':count}
    torch.save(data,out/f'layer{layer:02d}_window{count:02d}.pt')
    self._activation_counts[layer]=count+1
   return capture
  selected=[]
  for name,module in self.model_runner.model.named_modules():
   if module.__class__.__name__=='Qwen3MoeSparseMoeBlock':
    layer=int(name.split('layers.')[1].split('.')[0])
    if layer in LAYERS:
     selected.append(name);self._activation_handles.append(module.register_forward_pre_hook(hook(name,layer)))
  assert len(selected)==len(LAYERS),selected
  return selected
 def capture_status(self):return self._activation_counts

def main():
 import torch,torch_npu
 from vllm import LLM,SamplingParams
 from tests.ppl_data import get_test_loader
 out=Path('tests/artifacts/activation_outliers_20261120').resolve()
 out.mkdir(parents=True,exist_ok=True)
 data=get_test_loader('wikitext2',seqlen=512,model='/storage2/Qwen3-30B-A3B/').input_ids
 windows=data[0,:8*512].reshape(8,512).tolist()
 manifest={'model':'/storage2/Qwen3-30B-A3B/','dataset':'WikiText2 raw test','windows':8,'tokens_per_window':512,'layers':sorted(LAYERS),'input_sha256':hashlib.sha256(torch.tensor(windows,dtype=torch.int64).numpy().tobytes()).hexdigest(),'physical_npu':1,'tp':1,'model_dtype':'BF16','quantization_probe':'torch_npu.npu_dynamic_quant, per-token INT8 activation; BF16 model unchanged','scope':'MoE input before router/dispatch, prefill only; not GMM2 input or full W8A8/PPL evaluation'}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 llm=LLM(model=manifest['model'],dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,max_model_len=513,max_num_batched_tokens=513,max_num_seqs=1,gpu_memory_utilization=.99,block_size=128,num_gpu_blocks_override=8,enable_prefix_caching=False,enable_chunked_prefill=False,seed=0,worker_extension_cls='tests.performance.capture_activation_outliers.ActivationCaptureExtension')
 print('HOOKS',llm.collective_rpc('install_activation_capture',args=(str(out/'tensors'),)),flush=True)
 for i,w in enumerate(windows):
  outputs=llm.generate([{'prompt_token_ids':w}],SamplingParams(temperature=0,max_tokens=1,ignore_eos=True),use_tqdm=False)
  assert len(outputs)==1
  print('CAPTURED_WINDOW',i,flush=True)
 status=llm.collective_rpc('capture_status');(out/'capture_status.json').write_text(json.dumps(status,indent=2));print('COMPLETE',status,flush=True)
if __name__=='__main__':main()
