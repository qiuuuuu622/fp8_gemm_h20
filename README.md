# Replacing DeepGEMM for SGLang block-FP8 dense linears on H20

**Goal.** On the PD prefill node, SGLang serves `/prefill_model` (Qwen3.6-35B-A3B, 128×128 block FP8). Its dense
FP8 linears run through DeepGEMM, and DeepGEMM JIT-precompiles every M from 1 to 16384 at startup, which makes
each release slow to come up. This repo replaces that path with Triton/Gluon kernels and a drop-in SGLang patch.
The patch is opt-in via one environment variable and is removed by unsetting it.

Work followed the [NVlabs/kda](https://github.com/NVlabs/kda) loop (contract → candidate → validate → record).
Every candidate, including the rejected ones, is in `candidates.jsonl`.

## Results (1× H20, SGLang 0.5.20, same server args, A/B = DeepGEMM vs ours)

| | DeepGEMM | ours | |
|---|---|---|---|
| **Cold start**, empty JIT caches | 334.2 s | **173.6 s** | −48% |
| **Warm start**, caches reused (2 runs) | 183.6 / 178.6 s | **117.7 / 114.2 s** | −36% |
| Prefill CUDA-graph capture, cold → warm | 247.5 → 101.6 s | 79.9 → 37.8 s | DeepGEMM's per-M warmup happens here |
| Kernel-only cold compile, 6 shapes | 207.5 s (all M) / 143.9 s (`FAST_WARMUP`) | **17.0 s** | `runs/startup_cost_*.json` |
| **Prefill TTFT**, 8192 tokens, CUDA graphs on | 295.3 ms | **268.8 ms** | **1.10×** |
| Teacher-forced NLL, 4 texts / 348 tokens | 0.6580 | 0.6618 | +0.58% relative |

`runs/startup_ready.txt` holds the startup numbers. The model was copied to `/dev/shm` so ceph read speed doesn't
dominate the timing. The SGLang launch args are those of the prefill node, minus dynamo, PD and speculative
decoding.

### End-to-end A/B (`sgl_patch/ab_e2e.py`, `runs/ab_e2e_round{1,2}*.json`)

Both servers run side by side with warm caches, a breakable prefill CUDA graph and a full decode CUDA graph.
Every request uses unique random tokens, so the radix cache can't reuse prefixes. Each case is the median of 3 runs
with the two servers alternating order. Run 2 swaps the GPUs to rule out a card-to-card difference.

| case | TTFT DG | TTFT ours | speedup, run 1 / run 2 (GPUs swapped) | other |
|---|---|---|---|---|
| prefill 1024 tok | 51.1 ms | 48.6 ms | 1.052 / 1.040 | |
| prefill 4096 | 150.2 | 138.3 | 1.086 / 1.088 | |
| **prefill 8192** (= chunk size) | 295.3 | 268.8 | **1.098 / 1.096** | |
| prefill 16384 | 666.7 | 612.7 | 1.088 / 1.080 | |
| prefill 32768 | 1658.9 | 1546.0 | 1.073 / 1.063 | |
| 8192 tok × 4 concurrent | 1143.7 | 1031.0 | 1.109 / 1.098 | req/s 3.42 → 3.78 |
| 2048 in / 128 out | 82.1 | 76.2 | 1.077 / 1.074 | TPOT 3.63 → 3.15 ms |
| 8192 in / 128 out | 296.4 | 268.2 | 1.105 / 1.092 | TPOT 3.80 → 3.32 ms |
| 1024 in / 256 out × 16 conc | 349.8 | 318.8 | 1.097 / 1.092 | TPOT 8.37 → 7.77 ms |

**Why about 1.10× and not more.** Dense GEMMs are only part of prefill; MoE, GDN and attention make up the rest.
At M=8192 the dense GEMMs take 94.8 ms per forward with DeepGEMM and 84.2 ms with ours, a 1.126× gain on that
part alone (`sgl_patch/gemm_share.py`, layer counts 30 GDN + 10 full-attention). Activation quantization costs the
same on both paths. The TPOT rows don't apply to the PD prefill node, which doesn't decode, and these probes ran
without speculative decoding.

### Accuracy (`sgl_patch/compare_*.py`)

- **Kernel level:** `IMPL=dispatch python test_correct.py` passes all 585 cases, with worst relative error 7.5e-3
  against an fp32 dequantized reference. `test_c11_paths.py` passes 4032 c11 configs, plus CUDA-graph replay and
  two concurrent streams.
- **Patch level, real weights:** across every dense FP8 layer at M ∈ {7, 100, 333, 4096}, the worst relative
  difference from the DeepGEMM path is 1.05e-3 (`sgl_patch/test_patch.py`).
- **Server level:** each path is deterministic against itself (8/8 identical greedy outputs). Across paths, 2/8
  greedy continuations are identical. The other 6 diverge at a near-tie, where the gap between the top two
  logprobs is 0–0.5 nats, so they are rounding flips rather than errors. Mean per-token |Δlogp| is 0.002–0.10, and
  teacher-forced NLL differs by +0.58% relative. A downstream eval, such as GSM8K or an internal set, has **not**
  been run.

### Kernel-level comparison vs DeepGEMM (JIT excluded, `runs/bench_prefill_c11.csv`)

6 `/prefill_model` shapes × 20 M values, median of 3 × CUDA-graph timings:

| M range | cases | ours / DG (geomean) | min | max | DG faster |
|---|---|---|---|---|---|
| M ≤ 128 (decode-sized) | 42 | 1.171 | 0.802 | 2.102 | 18 |
| 128 < M ≤ 512 | 24 | 0.993 | 0.777 | 1.298 | 15 |
| 512 < M < 4096 | 24 | 1.129 | 0.847 | 1.644 | 4 |
| M ≥ 4096 (prefill chunks) | 30 | **1.196** | 1.020 | 1.357 | 0 |
| all | 120 | 1.131 | 0.777 | 2.102 | 37 |

The remaining losses sit at M values between tuning points (37, 64, 200, 777). The worst is o/out_proj at M=200,
at 0.78×. Densifying the dispatch grid would recover part of this. It's low priority for the prefill node, whose M
values cluster around the chunk size.

## Using it in SGLang (`sgl_patch/`)

The patch doesn't modify SGLang source. A `sitecustomize.py` on `PYTHONPATH` loads `fp8mm_sglang_patch.py`, which
does nothing unless `FP8MM_ENABLE=1` is set. When enabled, it wraps
`sglang.srt.layers.quantization.fp8_utils.dispatch_w8a8_block_fp8_linear`, and only when that function would have
returned the DeepGEMM linear. The replacement quantizes activations with SGLang's own
`sglang_per_token_group_quant_fp8`, using row-major fp32 scales, then calls `dispatch.matmul`.

Inputs fall back to the original DeepGEMM function when any of these holds:
- the (N,K) pair is not in `runs/dispatch.json`;
- the input is not bf16;
- the activations arrive pre-quantized;
- the block size is not [128, 128].

The patch requires `DEEPGEMM_SCALE_UE8M0 == False`. That flag is false on H20/Hopper, so weight scales stay plain
fp32 `[N/128, K/128]`.

```bash
# enable (prefill workers only)
export FP8MM_HOME=/path/to/fp8_gemm_h20          # dispatch.py, opt2.py, gl*.py, runs/dispatch.json
export PYTHONPATH=$FP8MM_HOME/sgl_patch:$PYTHONPATH
export FP8MM_ENABLE=1
export FP8MM_DISABLE_DG_PRECOMPILE=1             # sets SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 (the all-M warmup)
python3 -m dynamo.sglang ...                      # unchanged args
# rollback: unset FP8MM_ENABLE (or drop the PYTHONPATH entry)
```

To verify, look for `[fp8mm] patched ...dispatch_w8a8_block_fp8_linear` in the worker log. With the patch active,
no `DeepGEMM JIT Compiling for <GEMM_NT_F8F8BF16>` lines should appear.

Under `dynamo.sglang`, the patch has only been tested through `sglang.launch_server`, not through dynamo itself.
Because it's loaded via `sitecustomize`, it applies to any Python process that inherits `PYTHONPATH`, including
spawned scheduler processes.

Persist `SGLANG_CACHE_DIR` (Triton/Gluon JIT, inductor) across releases to get warm-start times. A cold cache
still compiles about 60 kernels, roughly 17 s of kernel compile work, instead of DeepGEMM's per-M warmup.

## Kernels

| name | file | what | where used |
|---|---|---|---|
| c3 | `opt2.py` | Triton `tl.dot`. When BLOCK_N divides 128, the weight scale is one scalar per K group, so promotion is a single FMA. | some 64–2048 M points |
| **c11** | `gl5.py` (+ `gl1.py`, `gl2.py`) | Gluon (Triton 3.7). TMA ring with mbarriers. **Pairwise async wgmma**: k and k+1 are issued together, then `wait(1)` → promote k → `wait(0)` → promote k+1, so no wgmma is in flight across the loop back-edge. Plus SWAP_AB for tiny M and tail-wave split-K with deterministic fixup and per-stream workspaces. | most points |
| c9 | `gl2.py` | c11's predecessor; still importable | not selected |

`dispatch.py` maps (N,K) and M to an impl and config from `runs/dispatch.json`, choosing the log-nearest tuned M.
`sweep_c11.py` rebuilds the table.

**Key finding (ncu + ptxas, `runs/ptxas_c3_c9.txt`, `runs/ncu_r1_summary.txt`).** c9's single-wgmma-in-flight loop
compiled with ptxas warning **C7514**: *"wgmma.mma_async instructions are serialized due to non wgmma instructions
reading accumulator registers"*. The PTX said `wait_group 1`, but the SASS was serialized; 27% of stall samples sat
on `WARPGROUP.DEPBAR`. c11's pairwise scheme removes C7514. This is what earlier pipeline-depth experiments (c10,
e1) couldn't fix, and why they failed. c11 moved 2048×4096 M=100 from 0.80× to 1.02× DeepGEMM, and 12288×2048
M=8192 from 1.01× to 1.11×.

**Rejected this round.** c12, a warp-specialized TMA-producer/MMA-consumer kernel via `gl.warp_specialize`: it lost
to c11 on 6 of 7 shapes, at 0.71–0.95×. Capping `maxnreg` at 128/168/232 didn't help either: 0.43–0.86×.

## Reproduce

```bash
# env: torch 2.13.0+cu130, triton 3.7.1, sglang 0.5.20, sgl-deep-gemm 0.2.0, CUDA 13.0, driver 550.54.15, H20
IMPL=dispatch python test_correct.py && python test_c11_paths.py           # correctness
python bench_prefill.py runs/bench_prefill_c11.csv                          # kernel vs DeepGEMM
python startup_cost.py {dg_full,dg_fast,ours}                               # kernel-only cold compile
sgl_patch/launch_probe.sh <tag> <gpu> <port> {dg,ours} {cold,warm}         # server startup (MODEL=/path)
python sgl_patch/ab_e2e.py <port_dg> dg <port_ours> ours out.json           # e2e A/B
python sgl_patch/compare_gen.py <p_dg> <p_ours>; python sgl_patch/compare_nll.py <p_dg> <p_ours>
```

## Caveats

- **Single host:** one H20 node (2 GPUs used), synthetic random-token prompts, ≤ 32k-token prefills. Production
  allows 262k context and a real traffic mix; neither is measured.
- **Startup is measured with `sglang.launch_server`**, not `dynamo.sglang`, and without NIXL/PD. The DeepGEMM
  saving is in-process, so it should carry over, but that's unverified.
- **No downstream task eval** has been run, only NLL and greedy-divergence checks.
- **Eager-mode launch overhead:** the Gluon path costs about 65 µs of Python per call against DeepGEMM's
  about 20 µs. That's hidden under CUDA graphs, which SGLang uses for both prefill (breakable) and decode, and it's
  visible only in eager mode.
- **`DEEPGEMM_SCALE_UE8M0=True` is not supported** (Blackwell). The patch then leaves DeepGEMM in place, because
  the scales arrive in a packed layout.

---

# History: v1 (2026-10-14, h20-4)

The original kernel-only study (c1–c10) follows. For per-candidate notes see `candidates.jsonl`.

W8A8 block-wise FP8 GEMM (`C[M,N] = A[M,K] · B[N,K]ᵀ`, bf16 out), activation scales 1×128 per token,
weight scales 128×128 per block — the `_w8a8_block_fp8_matmul_hopper` path used by SGLang dense FP8
linears. Optimized for the dense layers of Qwen3.6-35B-A3B prefill on H20 (78 SMs), following the
[NVlabs/kda](https://github.com/NVlabs/kda) workflow (contract → draft → candidate → validate → record).

### Result

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

### Kernels

`dispatch.matmul(A, B, As, Bs)` looks up `runs/dispatch.json`, which maps each (N,K) and the nearest M on a
log scale to a kernel and config, then runs:

| name | file | what changed vs the original `_w8a8_block_fp8_matmul_hopper` |
|---|---|---|
| **c3** | `opt2.py` | Triton `tl.dot`. When BLOCK_N divides 128, the weight scale is one scalar per K group, so promotion is a single `acc += dot * (a_s*b_s)[:,None]`. Per-shape tuned configs. Used for 118/120 test points. |
| **c9** | `gl2.py` (+`gl1.py`) | Gluon (Triton 3.7). Explicit TMA ring buffer with mbarriers, async `warpgroup_mma` with `wait_group 1` so scale promotion of step k overlaps wgmma k+1, SWAP_AB for tiny M (tokens on the MMA N side), and split-K on the tail wave only with deterministic fixup and per-stream workspaces. Wins at M ≤ 32. |

`baseline.py` holds the original kernel, used as the reference for speedups.

Input contract: A, B row-major fp8 e4m3 and contiguous; `As` fp32 `[M, K/128]`; `Bs` fp32 `[N/128, K/128]`; `K % 128 == 0`.

### Shapes

Qwen3.6-35B-A3B dense FP8 linears (`shapes.py`), TP=1: `gdn.in_proj_qkvz` 12288×2048, `gdn.out_proj` /
`attn.o_proj` 2048×4096, `attn.qkv_proj` 9216×2048, `shared.gate_up` 1024×2048, `shared.down` 2048×512,
plus TP=2/4 splits. Routed experts use the fused-MoE kernel and are out of scope.

### Correctness

- `IMPL=dispatch python test_correct.py`: **PASS**, 585 cases. They cover 15 (N,K) × 13 M (including M=1, 7, 31, 333, 4097, 8191)
  × 3 input distributions: normal, activation outliers, and skewed weight scales. Worst relative error vs reference is 7.5e-3.
  Worst difference from the original kernel's output is 5.1e-3. `runs/test_correct_dispatch.log`.
- `python test_c9_paths.py`: **PASS**. 2016 c9 configs (every SWAP_AB × SPLIT × tile combo), ragged M, a bit-exact check
  on a repeated call to confirm the counters reset, CUDA-graph replay ×50, and two concurrent streams running split-K ×200.

### Reproduce

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

### Optimization log

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

### Why DeepGEMM still wins at small M

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

### Caveats

- **Dispatch table vs test points:** the table was tuned at M ∈ {16, 32, 64, 128, 192, 256, 384, 512, 1024, …}. The test M
  values 100 and 333 use the nearest tuned entry, so small-M results are slightly pessimistic for c9.
- **Data:** inputs are synthetic Gaussian data quantized per block, not real activations.
- **Timing:** single-GPU, CUDA-graph timing, with no concurrent kernels and no end-to-end model measurement.


