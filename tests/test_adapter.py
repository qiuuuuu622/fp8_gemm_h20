"""Adapter contract checks without installing SGLang or initializing CUDA."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch, Mock

class Tensor:
    def __init__(self, shape, dtype='bf16', contiguous=True):
        self.shape=shape; self.dtype=dtype; self.contiguous=contiguous
    def is_contiguous(self): return self.contiguous
    def numel(self):
        import math
        return math.prod(self.shape)
    def view(self,*shape):
        import math
        known=math.prod(x for x in shape if x != -1)
        return Tensor(tuple(self.numel()//known if x==-1 else x for x in shape),self.dtype)
    def __add__(self,other): return self

class Adapter(unittest.TestCase):
    def setUp(self):
        self.saved=list(sys.meta_path)
        self.env=patch.dict(os.environ,{'FP8MM_ENABLE':'1','FP8MM_PROFILE':'122b_tp4','FP8MM_HOME':str(Path(__file__).resolve().parents[1])})
        self.env.start(); self.path=list(sys.path)
        spec=importlib.util.spec_from_file_location('_adapter_test',Path(__file__).resolve().parents[1]/'sgl_patch/fp8mm_sglang_patch.py')
        self.adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.adapter)
        self.orig=Mock(return_value='fallback')
        self.mod=types.SimpleNamespace(deepgemm_w8a8_block_fp8_linear_with_fallback=self.orig,
            dispatch_w8a8_block_fp8_linear=lambda:self.orig,
            sglang_per_token_group_quant_fp8=Mock(return_value=('q','s')),DEEPGEMM_SCALE_UE8M0=False)
        self.torch=patch.dict(sys.modules,{'torch':types.SimpleNamespace(bfloat16='bf16',float32='f32',float8_e4m3fn='fp8')})
        self.torch.start();self.adapter._install(self.mod)
        self.linear=self.mod.dispatch_w8a8_block_fp8_linear()
    def tearDown(self):
        self.torch.stop(); self.env.stop();sys.meta_path[:]=self.saved;sys.path[:]=self.path
    def args(self):return [Tensor((200,2048)),Tensor((3072,2048),'fp8'),[128,128],Tensor((24,16),'f32')]
    def test_profile_is_forwarded(self):
        with patch('fp8_gemm_h20.matmul',return_value=Tensor((200,3072))) as fn:
            args=self.args()
            out=self.linear(*args)
        self.assertEqual(out.shape,(200,3072))
        fn.assert_called_once_with('q',args[1],'s',args[3],profile='122b_tp4')
        self.assertEqual(fn.call_args.kwargs,{'profile':'122b_tp4'})
        self.orig.assert_not_called()
    def test_unsupported_inputs_fall_back_before_quantization(self):
        for kind in ('shape','range','dtype','weight_dtype','scale_dtype','block','prequant','ue8','noncontiguous'):
            args=self.args();kw={}
            if kind=='shape':args[1]=Tensor((1024,2048),'fp8')
            if kind=='range':args[0]=Tensor((16385,2048))
            if kind=='dtype':args[0].dtype='f32'
            if kind=='weight_dtype':args[1].dtype='bf16'
            if kind=='scale_dtype':args[3].dtype='bf16'
            if kind=='block':args[2]=[64,128]
            if kind=='prequant':kw['input_scale']=Tensor((200,16),'f32')
            if kind=='ue8':self.mod.DEEPGEMM_SCALE_UE8M0=True
            if kind=='noncontiguous':args[0].contiguous=False
            self.assertEqual(self.linear(*args,**kw),'fallback',kind)
            self.mod.DEEPGEMM_SCALE_UE8M0=False
        self.mod.sglang_per_token_group_quant_fp8.assert_not_called()
    def test_non_deepgemm_dispatch_is_unchanged(self):
        other=object();self.mod.dispatch_w8a8_block_fp8_linear=lambda:other
        self.adapter._install(self.mod)
        self.assertIs(self.mod.dispatch_w8a8_block_fp8_linear(),other)

if __name__=='__main__':unittest.main()
