# Benchmarks — MiniMax-M3-MXFP4 on 4× DGX Spark (GB10)

Config: this repo's `minimax-m3-mxfp4.yaml` (MXFP4, TP=4, **EAGLE3 k=2** — the best-performing
draft depth here, bf16 KV, block-size 128, NCCL 2.30.4, stock vLLM nightly `93d8f834`). Workload:
**prefill = 2048 tokens, decode = 512 tokens**, 3 runs averaged per row.

Run with:

```bash
sparkrun benchmark --rootful --skip-run ./minimax-m3-mxfp4.yaml \
  -b pp=2048 -b tg=512 -b depth=0,4096,16384,32768,65536 -b concurrency=1,2,5 -b runs=3
```

(`--skip-run` benchmarks an already-running server.)

- **depth** — pre-existing context depth (KV already filled) before the measured request
- **conc** — concurrent requests
- **pp t/s** — prefill (prompt) throughput, tokens/s
- **tg t/s** — generation (decode) throughput, tokens/s (aggregate across `conc`)
- **ttfr ms** — time to first response, ms

| depth |  conc | pp t/s | tg t/s | ttfr ms | runs |
|------:|------:|-------:|-------:|--------:|-----:|
|     0 |     1 | 2020.2 |   34.8 |  1015.1 |    3 |
|     0 |     2 | 2046.7 |   50.3 |  1652.2 |    3 |
|     0 |     5 | 2154.1 |   70.1 |  3536.3 |    3 |
|  4096 |     1 | 1726.0 |   34.2 |  1187.8 |    3 |
|  4096 |     2 | 1622.3 |   50.7 |  2045.4 |    3 |
|  4096 |     5 | 1755.8 |   69.2 |  4127.0 |    3 |
| 16384 |     1 | 1552.0 |   33.2 |  1331.3 |    3 |
| 16384 |     2 | 1647.3 |   45.2 |  1867.0 |    3 |
| 16384 |     5 | 1700.8 |   63.9 |  4337.6 |    3 |
| 32768 |     1 | 1557.6 |   30.8 |  1316.3 |    3 |
| 32768 |     2 | 1565.2 |   44.7 |  1966.5 |    3 |
| 32768 |     5 | 1683.8 |   58.8 |  4329.9 |    3 |
| 65536 |     1 | 1383.3 |   28.6 |  1482.3 |    3 |
| 65536 |     2 | 1318.6 |   39.9 |  2403.2 |    3 |
| 65536 |     5 | 1358.2 |   49.7 |  5232.2 |    3 |

## Takeaways

- **Single-stream decode ≈ 35 tok/s** at low context — roughly **4× the chthonic/b12x NVFP4
  path (~8 tok/s)** on the same hardware. This is the whole reason for the MXFP4 + stock-vLLM
  route (Marlin MoE + Triton sparse attention have real batch=1 decode paths on GB10).
- **Prefill ≈ 1.4-2.2k tok/s**, easing gently with depth.
- **Decode scales with concurrency:** ~35 → 50 (2×) → 70 (5×) tok/s aggregate at depth 0. Per-
  request decode at conc=5 is ~14 tok/s, so throughput-oriented serving benefits from raising
  `max_num_seqs` past the current 5 (KV permitting).
- **Context cost:** single-stream decode eases ~35 → 29 tok/s from 0 → 64K depth; prefill
  ~2020 → 1383. Long context is usable, just proportionally slower.
- **TTFR** grows with concurrency and depth (queuing + longer prefill), as expected.

> Numbers are decode-heavy MoE on GB10's unified memory; they will vary with EAGLE3 acceptance
> (temperature-sensitive) and `max_num_seqs` / `max_num_batched_tokens`.
