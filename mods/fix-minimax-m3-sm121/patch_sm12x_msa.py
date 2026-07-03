#!/usr/bin/env python3
"""Patch sparse_attention_msa.py to use fmha_sm12x for SM121.

fmha_sm12x.build_k2q_csr is the Torch reference (no return_schedule=True).
We need SparseK2qCsrBuilderSm12x (callable class) for the fused path.
"""
import sys, pathlib, site

_site = [p for p in site.getsitepackages() if "dist-packages" in p or "site-packages" in p][0]
path = pathlib.Path(_site) / "vllm/models/minimax_m3/nvidia/sparse_attention_msa.py"
if not path.exists():
    print("[patch_sm12x_msa] sparse_attention_msa.py not found; skipping")
    sys.exit(0)

content = path.read_text()

# Replace the import block and build_k2q_csr usage
old_import = """from vllm.third_party.fmha_sm100.sparse import (
                build_k2q_csr,
                sparse_atten_func,
            )"""
new_import = """from fmha_sm12x import (
                SparseK2qCsrBuilderSm12x,
                sparse_atten_func,
            )
            build_k2q_csr = SparseK2qCsrBuilderSm12x(use_optimized=True)"""

if old_import in content:
    content = content.replace(old_import, new_import)
    path.write_text(content)
    print("[patch_sm12x_msa] Patched sparse_attention_msa.py: fmha_sm100 -> fmha_sm12x (with SparseK2qCsrBuilderSm12x)")
elif "SparseK2qCsrBuilderSm12x" in content:
    print("[patch_sm12x_msa] already patched")
else:
    print("[patch_sm12x_msa] pattern not found; skipping")
