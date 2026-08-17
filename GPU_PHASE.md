# GPU phase guidelines — M5/M6/M7, with the C++ tracks

Implementation is yours; this file fixes scope, order, gates, and pitfalls.
Rule of thumb throughout: **every performance claim comes from a measurement
you can re-run** (plan §9.5) — three implementations and a table beat one
clever kernel every time.

## 0 · Entry gate (do these before any CUDA)

- [ ] git commit everything outstanding (features, baselines, Wyscout layer).
- [ ] E1 harness runnable (retrieval scoreboard exists to plug models into).
- [ ] VAEP per-action values on disk (the M5 quality head trains against them;
      also fills the quality axis for Wyscout entities, which have no xG).
- Sanity: `python -c "import torch; print(torch.cuda.get_device_capability())"`
  → (8, 9); bf16 supported.

## 1 · M5 — behaviour encoder (pure PyTorch; do NOT reach for C++ here)

Order of work, each with a done-when:

1. **Tensors.** (s, c) pairs from spadl: state = location, context, previous k
   actions, entity index; choice = destination cell on a 12×8 grid ×3 heights
   (passes/carries) or action-type class. Whole set `.to("cuda")` once.
   *Done when:* tensors resident, no DataLoader anywhere.
2. **Encoder v1.** Embedding table for entities + MLP policy head, CE loss.
   bf16 autocast, TF32 on, fused AdamW, `zero_grad(set_to_none=True)`.
   *Done when:* loss falls; a fixed query entity's neighbours in embedding
   space are not garbage.
3. **Throughput gate.** `nvidia-smi dmon -s u` while training, mains power,
   thermally steady. *Gate:* ≥70% sustained util BEFORE touching model size;
   if below, the input path is the bug (plan §9.3).
4. **InfoNCE regulariser** over disjoint action-bag halves within an entity;
   hard negatives = same pos_group + minutes band.
5. **GRL quality head** predicting per-action VAEP through gradient reversal.
   *The deliverable is the pair of numbers:* adversary accuracy before vs
   after — that's the style/quality separation evidence.
6. **Infra.** Hydra-or-argparse configs, MLflow local, checkpoint-and-resume
   (kill a run, resume it, verify metrics continue), GPU util logged as a
   metric, seeds ×3.
7. **Overnight ablations.** ±InfoNCE, ±GRL; mean±std over 3 seeds; scored on
   E1/E3/E4 like every other model. Expect to lose to M1 at first (plan
   budgets 2–3 rounds of choice-discretisation debugging).

## 2 · M6 C++ track — a CUDA kernel for freeze-frame geometry

Prereq: fetch 360 data (`git sparse-checkout add data/three-sixty` in the
statsbomb clone), stage frames as a `(N, 22, 2)` float tensor + mask.

**The exercise:** one geometry op (start with opponents-within-radius per
frame; line-breaking count is the stretch goal), implemented three ways:

- **(a) Python loop** over frames — baseline; measure on a 1k-frame sample
  only, extrapolate, never run in full.
- **(b) Batched torch** — broadcast `(B, 22, 2)`, `cdist`/norms + masking.
  This is the production path the plan expects.
- **(c) Your CUDA kernel** via `torch.utils.cpp_extension.load_inline` (start
  there; graduate to a `setup.py` build later): one block per frame or one
  thread per (frame, defender), reduce in shared memory.

What you'll actually learn, in the order you'll hit it: thread/block/grid
mapping → coalesced vs strided reads → shared-memory reduction →
`TORCH_CUDA_ARCH_LIST=8.9` build flags → pybind11 module boundary →
benchmarking discipline (`torch.cuda.synchronize()` around timers, warmup
iterations, thermally steady laptop).

**Gates:**
- Correctness: `torch.allclose(kernel_out, torch_out, atol=1e-5)` over random
  masked inputs, in a pytest.
- Benchmark table: frames/sec for (a)/(b)/(c) at 3 batch sizes.
- `ncu` on your kernel: memory- or compute-bound verdict + one guided
  optimisation iteration with before/after numbers.
- Writeup honesty: if (b) beats (c) — likely — say so and explain from the
  roofline; that analysis IS the profiling appendix.

**Environment (WSL2 — these bite):** install the `wsl-ubuntu` CUDA toolkit
variant ONLY (the generic `ubuntu2204` repo pulls a Linux driver and breaks
GPU passthrough — plan §1.1 rules 1–2); nvcc must match torch's CUDA major
(cu128 wheels → 12.x toolkit); ncu counter permissions are granted in the
WINDOWS NVIDIA Control Panel (Developer Settings → allow performance counters
to all users), not via Linux module params.

## 3 · M7 C++ track — multithreaded k-NN engine

**The exercise:** brute-force top-k cosine over the entity matrix (~21k × 66
now; z-scored floats), as a small C++ library:

1. Single-thread scalar baseline; verify against a numpy reference on the
   real parquet (read it via Arrow or dump to raw floats first).
2. Let the compiler vectorise: `-O3 -march=native -ffast-math`, check with
   `-fopt-info-vec`; measure the delta. (AVX2 on this machine.)
3. **OpenMP** over queries (embarrassingly parallel) and, separately, over
   rows with per-thread partial top-k heaps merged at the end — measure both,
   understand why one scales better (false sharing, merge cost).
4. Thread-scaling curve 1→12 threads; identify the knee and explain it
   (memory bandwidth, not cores).
5. pybind11 binding; swap it into the neighbours query path behind a flag.
6. Final table: your C++ (1 and N threads) vs numpy BLAS vs hnswlib, latency
   and recall@10. Expected conclusion: brute force is instant at this scale
   and ANN is unnecessary — now proven with your own implementation.

## 4 · Cross-cutting rules

- Determinism for reported numbers: seeds + `torch.use_deterministic_algorithms(True)`
  + `CUBLAS_WORKSPACE_CONFIG=:4096:8`; note the throughput cost.
- Laptop thermals: log `nvidia-smi -q -d TEMPERATURE,PERFORMANCE` next to any
  benchmark; numbers from a cold GPU are not comparable to steady state.
- OOM ladder: gradient accumulation → smaller batch →
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- CPU fallback (`--device cpu --subset <one comp>`) stays working in CI —
  reproducibility for GPU-less readers is a stated claim of the writeup.
- Every model lands on the same scoreboard (E1/E3/E4 + sanity suite), pooled
  and per-provider. No exceptions for the fancy ones.
