# MiniMax-M3 (MXFP4) on 4× DGX Spark / GB10

A **working, verified** recipe for serving `olka-fi/MiniMax-M3-MXFP4` (428B-A23B MoE,
sparse attention, vision) across **4× NVIDIA DGX Spark (GB10, sm_121, ARM64)** at
**tensor-parallel = 4**, over a switched **ConnectX-7 RoCE** fabric, with **EAGLE3**
speculative decoding — on **stock upstream vLLM nightly** launched
with [sparkrun](https://github.com/spark-arena/sparkrun).

## Why this stack

MiniMax-M3 is now in **upstream vLLM**. On GB10 the MXFP4 MoE runs through vLLM's **Marlin**
expert kernels and the sparse attention through **Triton** — which is what makes decode
usable here. (The chthonic/b12x path forces a throughput MoE kernel that is decode-inefficient
at batch=1 on GB10; and NVFP4 has no working alternative MoE kernel for sm_121.) So: **MXFP4 +
stock vLLM nightly + a small mod = the fast, correct path.**

## Performance

**~35 tok/s single-stream decode**, ~70 tok/s aggregate at
concurrency 5, ~2000 tok/s prefill. Full table (by context depth / concurrency) in
[`BENCHMARKS.md`](BENCHMARKS.md).

## Contents

| File | What |
|---|---|
| `minimax-m3-mxfp4.yaml` | the sparkrun recipe (the config that works) |
| `Dockerfile.m3-sm121` | `vllm/vllm-openai:nightly` **pinned** to `93d8f834…` + SM12x patches + NCCL 2.30.4 + baked SM121 fixes |
| `mods/fix-minimax-m3-sm121/` | the SM121 fixes (`run.sh` + files) — **baked into the image** at build time; kept here as the source |
| `BENCHMARKS.md` | measured prefill/decode throughput by depth & concurrency |

## Quick start

```bash
# 0) one-time sparkrun cluster setup (SSH mesh, CX7 NIC detection, static IPs, MTU 9000)
uvx sparkrun setup                       # add all 4 nodes

# 1) build the pinned image — bakes in NCCL 2.30.4 AND the SM121 fixes (runs run.sh at build
#    time). Build on aarch64 from THIS repo root so the `mods/` COPY resolves. Distribute / push.
docker build --no-cache -t vllm-m3-mxfp4-sm121:latest -f Dockerfile.m3-sm121 .

# 2) run it — NO `--rootful` (the fixes are baked, so the container runs rootless).
#    sparkrun downloads + distributes the target model on first run; the EAGLE3 draft
#    auto-downloads on each node at load time (needs internet to huggingface.co).
sparkrun run minimax-m3-mxfp4.yaml --tp 4
sparkrun status
sparkrun logs  <id>
sparkrun stop  <id>
```

Bring-up is ~8-12 min (per-node ~58 GB shard load + compile + warmup).

## The SM121 fixes (`mods/fix-minimax-m3-sm121/run.sh`)

**Baked into the image at build time** — the Dockerfile runs `run.sh` during `docker build`, so
nothing is patched at container start and the container runs rootless. (The `mods/` dir stays in the
repo as the source of these patches; it's `COPY`d in and executed, then removed.) Steps:

| # | Fix | Essential? |
|---|---|---|
| 0-1 | CUDA dev headers + `fmha_sm12x` install | yes (image plumbing) |
| 2 | `select_main_impl_cls` → official Triton sparse-attn impl on SM121 | yes |
| 3 | (skip the custom CUDA MSA — Triton fallback) | — |
| 4-5 | MXFP8 MoE oracle EMULATION + MARLIN assert (for the MXFP8 sibling model) | for MXFP8 |
| 6 | skip startup `reset_mm_cache` (multi-node NCCL hang) | yes |
| **7** | **MXFP4 MoE SwiGLU-OAI clamp** — forwards `gemm1_clamp_limit` | **REQUIRED for MXFP4** |

Step **7** is the one MXFP4 can't boot without (otherwise: `AssertionError: SWIGLUOAI_UNINTERLEAVE
requires clamp_limit`); credit [`0xSero/minimax-m3-sm120`](https://github.com/0xSero/minimax-m3-sm120)
for it. Everything else is the nightly's own code — **do not** overlay 0xSero's `sparse_attention.py`/
`indexer.py`: they're built for a different vLLM and break this nightly's model↔indexer contract
(`TypeError: MiniMaxM3Indexer.__init__() got an unexpected keyword argument 'topk_indices_buffer'`).
The stock sparse path works, and the multi-node hang was NCCL 2.28.9, fixed by baking in 2.30.4.

## Gotchas / hard-won learnings

These each cost a bring-up to find — they'll bite again on the next image bump:

1. **NCCL 2.30.4 is mandatory.** The bundled **2.28.9 hangs multi-node collectives on GB10** —
   the server starts, then the *first* generation dies with `RPC call to sample_tokens timed out`
   after a stalled all-reduce. Fixed by **baking NCCL 2.30.4 into the image** — `Dockerfile.m3-sm121`
   pulls the official `nvidia-nccl-cu13==2.30.4` aarch64 wheel into `/opt/nccl-2.30.4`, and the
   recipe LD_PRELOADs it. Confirm the banner reads `vLLM is using nccl==2.30.4`.
2. **Pin the nightly.** `vllm/vllm-openai:nightly` moves daily; only the pinned SHA
   (`93d8f834…`) is validated. Bump deliberately and re-test the whole stack.
3. **MXFP4 SwiGLU clamp (mod step 7).** The nightly's compressed-tensors MXFP4 MoE drops the
   model's `swiglu_limit=7.0`, so the Marlin activation asserts on a missing `clamp_limit`.
4. **Clear the image ENTRYPOINT.** The base image ships `ENTRYPOINT ["vllm","serve"]`; sparkrun
   runs `<image> bash -c "<cmd>"`, so without clearing it you get `vllm serve bash -c …` and the
   head never starts. Handled by `ENTRYPOINT []` in the Dockerfile **and** `executor_config.entrypoint: ""`.
5. **Run rootless (no `--rootful`).** The SM121 fixes are baked into the image at build time, so
   the container never patches `site-packages` at runtime and doesn't need root. Rootless keeps the
   HF cache owned by your user. Running **`--rootful` writes root-owned `.no_exist` negative-cache
   markers** into the shared HF cache, and sparkrun's next model-distribution rsync (run as your SSH
   user) then dies with `rsync … failed to set times … Operation not permitted (rc=23)`. If you ever
   did run rootful, fix it with `sudo chown -R "$(id -u):$(id -g)" ~/.cache/huggingface` on every node.
   (The recipe still points all caches at the writable `/cache/huggingface` mount.)
6. **No fp8 KV cache.** `--kv-cache-dtype fp8` forces FlashInfer, which needs page < 128 while
   M3's sparse attention **requires block 128** → `No common block size for 128`. It's also
   incorrect past ~2K tokens on sm121. Keep **bf16 (auto) KV** and **`--block-size 128`**.
7. **TP=4 = no padding.** 64 Q heads / 4 KV heads both divide by 4 cleanly.
8. **Keep disk clear.** `/tmp/ray` fills fast; a 95%-full disk triggers Ray spill warnings and
   can stall under load.
9. **sparkrun forces `HF_HUB_OFFLINE=1` by default.** So the EAGLE3 draft (not synced) fails with
   `Invalid repository ID … Inferact/MiniMax-M3-EAGLE3`. Set **`HF_HUB_OFFLINE: "0"` +
   `TRANSFORMERS_OFFLINE: "0"` explicitly** in the recipe env (omitting them isn't enough — the
   recipe value overrides sparkrun's default). Confirm the log no longer says `HF_HUB_OFFLINE is True`.

## Verify

```bash
# list models (HTTPS + api-key if you enabled them; -k for self-signed cert)
curl -sk https://<head-ip>:8000/v1/models | python3 -m json.tool

# generate
curl -sk https://<head-ip>:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"olka-fi/MiniMax-M3-MXFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

## Notes

- **Speculative decoding:** EAGLE3 draft `Inferact/MiniMax-M3-EAGLE3`, `num_speculative_tokens: 2`,
  `draft_tensor_parallel_size: 1`, `attention_backend: TRITON_ATTN`. **k=2 benchmarked best** here
  (k=3 draft-overhead outweighed the extra accepts); acceptance is temperature-sensitive (higher at
  low temp), so re-tune the token count if your traffic differs.
  **The draft is not synced by sparkrun** — the recipe runs non-offline so each node auto-downloads
  it at load time (needs internet to huggingface.co). Pin `"revision"` in the spec-config for
  reproducibility, or `huggingface-cli download Inferact/MiniMax-M3-EAGLE3` on each node for air-gapped runs.
- **Context/concurrency:** at `max_model_len: 262144` you trade single-request context for
  concurrency headroom — lower it for more `max_num_seqs` room, don't reach for fp8 KV.
- **MoE note:** the log warns "GPU does not have native support for FP4 … Marlin weight-only" —
  expected on GB10; decode is memory-bound and this is the fast path available here.

## Credits

This recipe stands on the work of the DGX Spark community:

- **[@ciprianveg](https://forums.developer.nvidia.com/u/ciprianveg)** on the NVIDIA DGX Spark
  forum — thread [*MiniMax-M3-NVFP4 and NVFP4-REAP-50% for 4x/2x DGX
  Sparks*](https://forums.developer.nvidia.com/t/minimax-m3-nvfp4-and-nvfp4-reap-50-for-4x-2x-dgx-sparks/373177/38).
  Source of the `vllm-m3-mxfp4-sm121` image approach (pinned nightly + SM12x patches) and the
  `fix-minimax-m3-sm121` mod (MSA install, sparse-attn Triton path, MXFP8 oracle, NCCL 2.30.4).
- **[`0xSero/minimax-m3-sm120`](https://github.com/0xSero/minimax-m3-sm120)** — the MXFP4 MoE
  SwiGLU-OAI clamp fix (mod step 7).
- **[olka-fi/MiniMax-M3-MXFP4](https://huggingface.co/olka-fi/MiniMax-M3-MXFP4)** — the MXFP4 quant.
- **[Inferact/MiniMax-M3-EAGLE3](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3)** — the EAGLE3 draft.
- **[sparkrun](https://github.com/spark-arena/sparkrun)** — the multi-node launcher.
- **[vLLM](https://github.com/vllm-project/vllm)** (M3 upstream support) and **MiniMax** (the model).
