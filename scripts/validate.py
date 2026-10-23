"""Shared 35B/122B correctness, graph replay and optional migration timing.

Use CUDA_VISIBLE_DEVICES to reserve one physical GPU. --reference points to
an unmodified 35B repository or the parent of the frozen fp8_gemm_122b package.
"""
import argparse
import importlib
import json
import statistics
import sys
from pathlib import Path
import torch
import triton
from fp8_gemm_h20 import matmul, select
from fp8_gemm_h20.config import get_profile

MS_35 = [1,8,16,37,64,100,128,200,256,333,512,777,1000,2048,3000,4096,5000,8192,12000,16384]
MS_122 = sorted(set([1,7,16,32,37,64,100,128,192,200,256,333,384,512,768,777,1024,2048,4096,4097,8192,16384,
    15,17,31,33,63,65,127,129,193,255,257,383,385,511,513,1023,1025,2049,23,39,95,130,191,258,510,1026,47,79,194,509]))

def make(m,n,k,mode='normal'):
    a=torch.randn(m,k,device='cuda'); b=torch.randn(n,k,device='cuda')*.05
    if mode=='outlier': a[:,::127]*=100
    if mode=='skew': b*=torch.logspace(-3,1,n//128,device='cuda').repeat_interleave(128)[:,None]
    if mode=='zero': a.zero_()
    sa=a.view(m,k//128,128).abs().amax(-1).clamp_min(1e-6)/448
    bb=b.view(n//128,128,k//128,128); sb=bb.abs().amax((1,3)).clamp_min(1e-6)/448
    aa=(a.view(m,k//128,128)/sa[...,None]).reshape(m,k).to(torch.float8_e4m3fn)
    bq=(bb/sb[:,None,:,None]).reshape(n,k).to(torch.float8_e4m3fn)
    return aa,bq,sa.contiguous(),sb.contiguous()

def reference(args):
    a,b,sa,sb=args; m,k=a.shape; n=b.shape[0]
    return (a.float().view(m,k//128,128)*sa[...,None]).reshape(m,k) @ (b.float().view(n//128,128,k//128,128)*sb[:,None,:,None]).reshape(n,k).T

def error(out,ref):
    return ((out.float()-ref).abs().max()/ref.abs().max().clamp_min(1e-30)).item()

def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=['35b_tp1','122b_tp4'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--reference');p.add_argument('--timing',action='store_true')
    p.add_argument('--deepgemm',action='store_true',help='Also benchmark DeepGEMM, excluding scale-layout conversion')
    p.add_argument('--rounds',type=int,default=5);p.add_argument('--rep-ms',type=int,default=60)
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    old=None
    if a.reference:
        sys.path.insert(0,str(Path(a.reference).resolve()))
        old=importlib.import_module('dispatch').matmul if a.profile=='35b_tp1' else importlib.import_module('fp8_gemm_122b').fp8_gemm_122b_tp4
    config=get_profile(a.profile); ms=MS_35 if a.profile=='35b_tp1' else MS_122
    output=dict(profile=a.profile,device=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,accuracy=[],timing=[],graphs=[],strided_scales=[])
    reps={}
    for nk in config['entries']:
        n,k=map(int,nk.split(','))
        for m in ms:
            entry,_=select(a.profile,m,n,k);c=entry['config']
            key=(entry['impl'],c.get('SPLIT',c.get('SPLIT_K',1)),c.get('MODE',0),c.get('UNROLL',False),k)
            reps.setdefault(key,(m,n,k))
            for mode in ('normal','outlier','skew','zero'):
                torch.manual_seed(m+n+k+113);args=make(m,n,k,mode)
                ref=reference(args);out=matmul(*args,profile=a.profile);err=error(out,ref)
                assert torch.isfinite(out).all() and err<.01,(m,n,k,mode,err)
                parity=None
                if old is not None:
                    prior=old(*args);parity=torch.equal(out,prior)
                    assert parity,(m,n,k,mode,'migration output mismatch')
                output['accuracy'].append(dict(M=m,N=n,K=k,mode=mode,error=err,bitwise_equal=parity))
            if a.timing and m in (1,37,64,128,200,511,512,8192):
                funcs={'unified':lambda:matmul(*args,profile=a.profile)}
                # Regenerate nonzero values for timing; lambda captures this current args binding.
                args=make(m,n,k)
                if old: funcs['original']=lambda:old(*args)
                if a.deepgemm:
                    import deep_gemm
                    aa,bb,sa,sb=args; sadg=deep_gemm.get_mn_major_tma_aligned_tensor(sa)
                    dest=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
                    funcs['deepgemm']=lambda:deep_gemm.fp8_gemm_nt((aa,sadg),(bb,sb),dest)
                for fn in funcs.values(): fn()
                torch.cuda.synchronize();samples={name:[] for name in funcs}
                for rnd in range(a.rounds):
                    for name in list(funcs)[::1 if rnd%2==0 else -1]:
                        samples[name].append(triton.testing.do_bench_cudagraph(funcs[name],rep=a.rep_ms)*1000)
                output['timing'].append(dict(M=m,N=n,K=k,samples_us=samples,median_us={name:statistics.median(v) for name,v in samples.items()}))
            print('checked',a.profile,m,n,k,flush=True)
    for m,n,k in reps.values():
        streams=[torch.cuda.Stream(),torch.cuda.Stream()];args_list=[make(m,n,k),make(m,n,k)]
        refs=[reference(args) for args in args_list];graphs=[];outs=[]
        for stream,args in zip(streams,args_list):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):matmul(*args,profile=a.profile)
        torch.cuda.synchronize()
        for stream,args in zip(streams,args_list):
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):out=matmul(*args,profile=a.profile)
            graphs.append(g);outs.append(out)
        for _ in range(100):
            for stream,g in zip(streams,graphs):
                with torch.cuda.stream(stream):g.replay()
        torch.cuda.synchronize()
        errs=[error(out,ref) for out,ref in zip(outs,refs)]
        assert max(errs)<.01,(m,n,k,errs)
        output['graphs'].append(dict(M=m,N=n,K=k,errors=errs))
        # Exercise stride-aware scales, including the scale-TMA fallback.
        args=list(make(m,n,k))
        for idx in (2,3):args[idx]=args[idx].T.contiguous().T
        ref=reference(args);out=matmul(*args,profile=a.profile);err=error(out,ref)
        assert torch.isfinite(out).all() and err<.01,(m,n,k,'strided scales',err)
        output['strided_scales'].append(dict(M=m,N=n,K=k,error=err))
    output['status']='passed';output['worst_error']=max(r['error'] for r in output['accuracy'])
    Path(a.output).write_text(json.dumps(output,indent=2)+'\n')
    print('PASS',a.profile,len(output['accuracy']),output['worst_error'],flush=True)

if __name__=='__main__':main()
