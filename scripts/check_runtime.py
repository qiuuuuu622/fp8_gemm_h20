"""API checks that require two visible H20s; never selects physical GPU IDs itself."""
import json
from pathlib import Path
import torch
from fp8_gemm_h20 import matmul
from .validate import make, reference, error


def main():
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2, 'Expose only the two authorized GPUs'
    args=make(37,3072,2048)
    on_second=tuple(t.to('cuda:1') for t in args)
    torch.cuda.synchronize(1)
    assert torch.cuda.current_device()==0
    out=matmul(*on_second,profile='122b_tp4')
    assert out.device.index==1 and torch.cuda.current_device()==0
    with torch.cuda.device(1):
        err=error(out,reference(on_second))
    assert err<.01
    rejected=[]
    for kind in ('dtype','scale_shape','scale_dtype','mixed_device','noncontiguous','unsupported_shape'):
        aa=list(args)
        if kind=='dtype':aa[0]=aa[0].to(torch.bfloat16)
        if kind=='scale_shape':aa[2]=aa[2][:,:-1]
        if kind=='scale_dtype':aa[3]=aa[3].to(torch.bfloat16)
        if kind=='mixed_device':aa[1]=aa[1].to('cuda:1')
        if kind=='noncontiguous':aa[0]=aa[0].T.contiguous().T
        if kind=='unsupported_shape':aa=list(make(37,1024,2048))
        try:matmul(*aa,profile='122b_tp4')
        except ValueError:rejected.append(kind)
        else:raise AssertionError(kind)
    result=dict(status='passed',non_current_device_error=err,rejected_inputs=rejected)
    Path('runtime_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)

if __name__=='__main__':main()
