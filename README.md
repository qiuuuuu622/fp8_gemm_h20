# Block-FP8 GEMM on NVIDIA H20 — Triton/Gluon kernels vs DeepGEMM

W8A8 block-wise FP8 GEMM (`C[M,N] = A[M,K] · B[N,K]ᵀ`, bf16 out), activation scales 1×128 per token,
weight scales 128×128 per block — the `_w8a8_block_fp8_matmul_hopper` path used by SGLang dense FP8
linears. Optimized for the dense layers of Qwen3.6-35B-A3B prefill on H20 (78 SMs), following the
[NVlabs/kda](https://github.com/NVlabs/kda) workflow (contract → draft → candidate → validate → record).

## Result

Clean run, 120 cases (6 model shapes × 12 M + 8 TP-split shapes × 6 M), median of 3×CUDA-graph timing,
DeepGEMM warmed (JIT compile time **excluded**). `runs/bench_final.csv`.

| M range | cases | ours / DeepGEMM (geomean) | min | max | DeepGEMM faster | ours / original kernel |
|---|---|---|---|---|---|---|
| M = 100 | 14 | 0.901 | 0.697 | 1.025 | 12 | 1.556 |
| M = 256, 333 | 20 | 0.940 | 0.785 | 1.055 | 18 | 1.257 |
| 1000 ≤ M < 4096 | 40 | **1.116** | 0.898 | 1.296 | 2 | 1.049 |
| M ≥ 4096 | 46 | **1.176** | 1.031 | 1.500 | 0 | 1.058 |
| all | 120 | 1.079 | 0.697 | 1.500 | 32 | 1.136 |

Max relative error vs fp32 dequant reference: ours 7e-3, DeepGEMM 7e-3.

- **Large M (≥ 1000):** ours wins 84/86 points. For reference, cuBLAS per-tensor FP8 without block
  scales reaches 260–283 TFLOPS on the large shapes (`runs/ceiling_cublas.csv`).
- **Large-layer throughput (M ≥ 4096, qkvz/qkv/o/out):** ours 259–275 TFLOPS, DeepGEMM 205–257.
- **Small M (100–333):** DeepGEMM wins 30/34 points. The gap is largest when N is small (N 512–3072,
  M 100–333: 0.70–0.82×). It narrows to 0.88–0.93× for N=12288 or K=4096.

Tested M values are 100, 256, 333, 1000, 1024, 2048, 3000, 4096, 5000, 8192, 12000 and 16384, so there is no M=512 point.

### Where DeepGEMM is still faster (ours/DG < 0.95)

| layer | N | K | M | kernel | ours/DG | TFLOPS ours vs DG |
|---|---|---|---|---|---|---|
| sh_gu.tp2 | 512 | 2048 | 100 | c9 | 0.697 | 34.1 vs 48.9 |
| qkv.tp4 | 2304 | 2048 | 100 | c3 | 0.768 | 99.4 vs 129.5 |
| shared.gate_up | 1024 | 2048 | 333 | c9 | 0.785 | 121.2 vs 154.3 |
| sh_gu.tp2 | 512 | 2048 | 333 | c3 | 0.792 | 87.2 vs 110.1 |
| qkvz.tp4 | 3072 | 2048 | 100 | c3 | 0.815 | 113.3 vs 139.0 |
| shared.gate_up | 1024 | 2048 | 100 | c3 | 0.821 | 64.9 vs 79.1 |
| attn.o_proj | 2048 | 4096 | 100 | c3 | 0.879 | 120.6 vs 137.2 |
| gdn.out_proj | 2048 | 4096 | 256 | c3 | 0.889 | 182.1 vs 204.9 |
| gdn.out_proj | 2048 | 4096 | 100 | c3 | 0.893 | 119.5 vs 133.8 |
| sh_gu.tp2 | 512 | 2048 | 1000 | c3 | 0.898 | 159.5 vs 177.5 |
| gdn.in_proj_qkvz | 12288 | 2048 | 256 | c3 | 0.909 | 232.3 vs 255.6 |
| gdn.in_proj_qkvz | 12288 | 2048 | 100 | c3 | 0.922 | 177.8 vs 193.0 |
| attn.o_proj | 2048 | 4096 | 333 | c3 | 0.930 | 202.6 vs 217.9 |
| gdn.in_proj_qkvz | 12288 | 2048 | 333 | c3 | 0.932 | 210.4 vs 225.8 |
| gdn.out_proj | 2048 | 4096 | 333 | c3 | 0.937 | 202.5 vs 216.0 |
| shared.down | 2048 | 512 | 256 | c3 | 0.939 | 108.2 vs 115.3 |
| shared.down | 2048 | 512 | 333 | c3 | 0.942 | 115.9 vs 123.1 |
| out.tp4 | 2048 | 1024 | 333 | c3 | 0.945 | 154.0 vs 163.0 |
| qkv.tp2 | 4608 | 2048 | 333 | c3 | 0.945 | 192.8 vs 204.0 |
| out.tp2 | 2048 | 2048 | 333 | c3 | 0.947 | 185.0 vs 195.4 |
| qkvz.tp2 | 6144 | 2048 | 100 | c3 | 0.949 | 167.2 vs 176.1 |

Another 11 points fall between 0.95 and 1.00. The full list is in `runs/bench_final.csv`.

**Recommended production dispatch:** use these kernels for M ≥ 1000, and route M ≤ 333 to DeepGEMM.
The crossover between those two points was not measured.

## Kernels

`dispatch.matmul(A, B, As, Bs)` looks up `runs/dispatch.json`, which maps each (N,K) and the nearest M on a
log scale to a kernel and config, then runs:

| name | file | what changed vs the original `_w8a8_block_fp8_matmul_hopper` |
|---|---|---|
| **c3** | `opt2.py` | Triton `tl.dot`. When BLOCK_N divides 128, the weight scale is one scalar per K group, so promotion is a single `acc += dot * (a_s*b_s)[:,None]`. Per-shape tuned configs. Used for 118/120 test points. |
| **c9** | `gl2.py` (+`gl1.py`) | Gluon (Triton 3.7). Explicit TMA ring buffer with mbarriers, async `warpgroup_mma` with `wait_group 1` so scale promotion of step k overlaps wgmma k+1, SWAP_AB for tiny M (tokens on the MMA N side), and split-K on the tail wave only with deterministic fixup and per-stream workspaces. Wins at M ≤ 32. |

`baseline.py` holds the original kernel, used as the reference for speedups.

Input contract: A, B row-major fp8 e4m3 and contiguous; `As` fp32 `[M, K/128]`; `Bs` fp32 `[N/128, K/128]`; `K % 128 == 0`.

## Shapes

Qwen3.6-35B-A3B dense FP8 linears (`shapes.py`), TP=1: `gdn.in_proj_qkvz` 12288×2048, `gdn.out_proj` /
`attn.o_proj` 2048×4096, `attn.qkv_proj` 9216×2048, `shared.gate_up` 1024×2048, `shared.down` 2048×512,
plus TP=2/4 splits. Routed experts use the fused-MoE kernel and are out of scope.

## Correctness

- `IMPL=dispatch python test_correct.py`: **PASS**, 585 cases. They cover 15 (N,K) × 13 M (including M=1, 7, 31, 333, 4097, 8191)
  × 3 input distributions: normal, activation outliers, and skewed weight scales. Worst relative error vs reference is 7.5e-3.
  Worst difference from the original kernel's output is 5.1e-3. `runs/test_correct_dispatch.log`.
- `python test_c9_paths.py`: **PASS**. 2016 c9 configs (every SWAP_AB × SPLIT × tile combo), ragged M, a bit-exact check
  on a repeated call to confirm the counters reset, CUDA-graph replay ×50, and two concurrent streams running split-K ×200.

## Reproduce

```bash
# env used: torch 2.13.0+cu130, triton 3.7.1, sgl-deep-gemm 0.2.0, CUDA 13.0, driver 550.54.15, 2x H20
python test_correct.py                      # IMPL=dispatch for the dispatcher
python test_c9_paths.py
python sweep_unified.py runs/dispatch_0.json 0 2 &   # rebuild the dispatch table (2 GPUs, ~25 min)
python sweep_unified.py runs/dispatch_1.json 1 2     # then merge into runs/dispatch.json
python bench_final.py runs/bench_final.csv  # 120-case comparison (~25 min)
python analyze_final.py runs/bench_final.csv
```

`bench.py`'s `base_tuned` mode reads SGLang's H20 tuned-config directory (`CFG_DIR`), and the other modes don't need it.

## Optimization log

Every candidate is recorded in `candidates.jsonl`. Rejected ideas are kept so they aren't retried.

| candidate | idea | outcome |
|---|---|---|
| c1 | config sweep of the original kernel | no gain over default config |
| c2 | TMA descriptors + persistent grid (Triton) | rejected, 0.68–0.95× |
| **c3** | scalar weight scale + single-FMA promotion + per-shape tuning | **promoted**: 1.138× vs original on model shapes |
| c5 | manual K unroll ×2 | rejected; PTX still `wait_group 0`, so `tl.dot` serializes wgmma |
| c6 | Triton SWAP_AB + in-kernel split-K | superseded by c9 |
| c7 | loop-carried (skewed) promotion | rejected, 0.6–1.0×, 251 regs |
| c8 | Gluon TMA ring + async wgmma | superseded by c9 |
| **c9** | c8 + SWAP_AB + tail split-K | **promoted** for tiny M |
| c10 | scale loads prefetched 2 steps ahead | rejected; deeper pipelines get slower, so scale latency isn't the limit |
| e1 | several tiles per CTA, ring carried across tiles | rejected; time follows CTA count, so per-tile fixed cost isn't the limit |

## Why DeepGEMM still wins at small M

- **Ruled out by measurement:**
  - **Pipeline depth:** stages 3→8 only gets slower.
  - **Scale-load latency:** c10 didn't help.
  - **Per-tile fixed cost:** e1 didn't help.
  - **Register/smem limits:** a 16×64 tile at 8 stages uses 80 KB smem and 48 regs.
- **Static estimate (Little's law):** c9 keeps about ⅓ of DeepGEMM's bytes in flight per SM, and it relies on several CTAs
  per SM to hide latency. DeepGEMM runs one persistent CTA per SM with warp-specialized TMA-producer and MMA-consumer
  warpgroups, plus 7–16 stages, and picks non-power-of-2 `block_n` (80/160) to fill whole waves. Triton tiles must be
  powers of 2.
- **Not yet verified:** Nsight Compute counters were blocked in this container (`ERR_NVGPUCTRPERM`), so stall reasons
  weren't measured. The next structural step is a Hopper warp-specialized Gluon kernel.

## Caveats

- **Dispatch table vs test points:** the table was tuned at M ∈ {16, 32, 64, 128, 192, 256, 384, 512, 1024, …}. The test M
  values 100 and 333 use the nearest tuned entry, so small-M results are slightly pessimistic for c9.
- **Data:** inputs are synthetic Gaussian data quantized per block, not real activations.
- **Timing:** single-GPU, CUDA-graph timing, with no concurrent kernels and no end-to-end model measurement.

