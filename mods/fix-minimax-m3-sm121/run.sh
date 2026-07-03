#!/bin/bash
# run.sh — Comprehensive GB10 (SM121) fixes for MiniMax-M3
# 0. CUDA dev headers   1. MSA fmha_sm12x   2. select_main_impl_cls
# 3. (skip MSA)         4. MXFP8 EMULATION  5. MARLIN MXFP8 assert
# 6. api_server reset_mm_cache
# 7. MXFP4 MoE SwiGLU-OAI clamp            (0xSero/minimax-m3-sm120)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[fix-minimax-m3-sm121] Applying comprehensive GB10 SM121 fixes"
echo "[fix-minimax-m3-sm121] SCRIPT_DIR=$SCRIPT_DIR"
SITE_PACKAGES="$(python3 -c 'import site; print([p for p in site.getsitepackages() if "dist-packages" in p or "site-packages" in p][0])')"
echo "[fix-minimax-m3-sm121] SITE_PACKAGES=$SITE_PACKAGES"

# ============================================================================
# 0. Install CUDA dev headers for SM12x CUDA JIT compilation
# ============================================================================
echo ""; echo "[fix-minimax-m3-sm121] === Step 0: Install CUDA dev headers ==="
if [[ -f "$SCRIPT_DIR/cuda_headers.tar.gz" ]]; then
    tar xzf "$SCRIPT_DIR/cuda_headers.tar.gz" -C /usr/local/cuda/
    echo "[fix-minimax-m3-sm121] CUDA headers installed to /usr/local/cuda/include/"
elif [[ -f "$SCRIPT_DIR/cusparse.h" ]]; then
    mkdir -p /usr/local/cuda/include
    cp "$SCRIPT_DIR/cusparse.h" /usr/local/cuda/include/cusparse.h
    echo "[fix-minimax-m3-sm121] Only cusparse.h installed (partial fix)"
else
    echo "[fix-minimax-m3-sm121] WARNING: no CUDA headers found in mod dir"
fi

# ============================================================================
# 1. Install MSA fmha_sm12x from pre-cloned source
# ============================================================================
echo ""; echo "[fix-minimax-m3-sm121] === Step 1: Install MSA fmha_sm12x ==="
if python3 -c "import fmha_sm12x" 2>/dev/null; then
    echo "[fix-minimax-m3-sm121] fmha_sm12x already installed"
else
    if [[ -d "$SCRIPT_DIR/msa_source/fmha_sm12x" ]]; then
        cp -r "$SCRIPT_DIR/msa_source/fmha_sm12x" "$SITE_PACKAGES/"
        cp -r "$SCRIPT_DIR/msa_source/minimax_msa" "$SITE_PACKAGES/" 2>/dev/null || true
        echo "[fix-minimax-m3-sm121] fmha_sm12x installed via file copy"
    elif [[ -d /tmp/MSA/python/fmha_sm12x ]]; then
        cp -r /tmp/MSA/python/fmha_sm12x "$SITE_PACKAGES/"
        cp -r /tmp/MSA/python/minimax_msa "$SITE_PACKAGES/" 2>/dev/null || true
        echo "[fix-minimax-m3-sm121] fmha_sm12x installed via file copy"
    else
        echo "[fix-minimax-m3-sm121] ERROR: MSA source not found at $SCRIPT_DIR/msa_source/"
        exit 1
    fi
fi
python3 -c "import fmha_sm12x; print('[fix-minimax-m3-sm121] fmha_sm12x import OK')" 2>&1 || {
    echo "[fix-minimax-m3-sm121] ERROR: fmha_sm12x import failed after install"; exit 1; }

# ============================================================================
# 2. Patch select_main_impl_cls: ensure SM121+MXFP8 uses official Triton impl
# ============================================================================
echo ""; echo "[fix-minimax-m3-sm121] === Step 2: Patch select_main_impl_cls for SM121 ==="
python3 - <<'PY'
import pathlib, site, re
_site = [p for p in site.getsitepackages() if "dist-packages" in p or "site-packages" in p][0]
path = pathlib.Path(_site) / 'vllm/models/minimax_m3/common/sparse_attention.py'
if not path.exists():
    print('[fix-minimax-m3-sm121] sparse_attention.py not found; skipping'); raise SystemExit
text = path.read_text()
if 'SM121_FIX_APPLIED' in text:
    print('[fix-minimax-m3-sm121] select_main_impl_cls already fixed for SM121'); raise SystemExit
sm121_pattern = r'\n\s*# SM120/SM121 \(GB10 / DGX Spark\): use fmha_sm12x with Triton reference backend\..*?return MiniMaxM3SparseSM12xImpl\n'
text = re.sub(sm121_pattern, '\n', text, flags=re.DOTALL)
text = re.sub(r'\n\s*# SM121.*?\n', '\n', text)
text = re.sub(r'from vllm\.models\.minimax_m3\.nvidia\.sparse_attention_sm12x import.*?\n', '', text)
text = text.replace('def select_main_impl_cls(',
    '# SM121_FIX_APPLIED: removed custom SM121 branch, official logic handles SM121+MXFP8 via Triton\n'
    'def select_main_impl_cls(')
path.write_text(text)
print('[fix-minimax-m3-sm121] patched select_main_impl_cls -> official Triton impl')
PY

# ============================================================================
# 3. SKIP MSA (SM12x CUDA API incompatible; use Triton fallback)
# ============================================================================
echo ""; echo "[fix-minimax-m3-sm121] === Step 3: SKIP MSA (Triton fallback) ==="

# ============================================================================
# 4. Add EMULATION backend to MXFP8 oracle (MARLIN crashes on SM121)
# ============================================================================
echo ""; echo "[fix-minimax-m3-sm121] === Step 4: Patch MXFP8 oracle to add EMULATION backend ==="
python3 - <<'PY'
import pathlib, site
_site = [p for p in site.getsitepackages() if "dist-packages" in p or "site-packages" in p][0]
path = pathlib.Path(_site) / 'vllm/model_executor/layers/fused_moe/oracle/mxfp8.py'
if not path.exists():
    print('[fix-minimax-m3-sm121] mxfp8.py oracle not found; skipping'); raise SystemExit
text = path.read_text()
if 'EMULATION' in text and '"emulation"' in text:
    print('[fix-minimax-m3-sm121] MXFP8 oracle already patched for EMULATION'); raise SystemExit
text = text.replace(
'''_SUPPORTED_BACKENDS = (
    Fp8MoeBackend.FLASHINFER_TRTLLM,
    Fp8MoeBackend.DEEPGEMM,
    Fp8MoeBackend.MARLIN,
    Fp8MoeBackend.XPU,
)''',
'''_SUPPORTED_BACKENDS = (
    Fp8MoeBackend.FLASHINFER_TRTLLM,
    Fp8MoeBackend.DEEPGEMM,
    Fp8MoeBackend.MARLIN,
    Fp8MoeBackend.XPU,
    Fp8MoeBackend.EMULATION,
)''')
text = text.replace(
'''_BACKEND_NAME_MAP: dict[str, Fp8MoeBackend] = {
    "flashinfer_trtllm": Fp8MoeBackend.FLASHINFER_TRTLLM,
    "deep_gemm": Fp8MoeBackend.DEEPGEMM,
    "marlin": Fp8MoeBackend.MARLIN,
    "xpu": Fp8MoeBackend.XPU,
}''',
'''_BACKEND_NAME_MAP: dict[str, Fp8MoeBackend] = {
    "flashinfer_trtllm": Fp8MoeBackend.FLASHINFER_TRTLLM,
    "deep_gemm": Fp8MoeBackend.DEEPGEMM,
    "marlin": Fp8MoeBackend.MARLIN,
    "xpu": Fp8MoeBackend.XPU,
    "emulation": Fp8MoeBackend.EMULATION,
}''')
text = text.replace(
'''    return backend_to_kernel_cls(backend)''',
'''    if backend == Fp8MoeBackend.EMULATION:
        from vllm.model_executor.layers.fused_moe.experts.mxfp8_emulation_moe import (
            Mxfp8EmulationTritonExperts,
        )

        return [Mxfp8EmulationTritonExperts]
    return backend_to_kernel_cls(backend)''')
path.write_text(text)
print('[fix-minimax-m3-sm121] patched MXFP8 oracle -> added EMULATION backend')
PY

# ============================================================================
# 5. Fix MARLIN MoE assertion to accept MXFP8 dynamic
# ============================================================================
echo ""; echo "[fix-minimax-m3-sm121] === Step 5: Fix MARLIN MoE assertion for MXFP8 ==="
python3 - <<'PY'
import pathlib, site
_site = [p for p in site.getsitepackages() if "dist-packages" in p or "site-packages" in p][0]
path = pathlib.Path(_site) / 'vllm/model_executor/layers/fused_moe/experts/marlin_moe.py'
if not path.exists():
    print('[fix-minimax-m3-sm121] marlin_moe.py not found; skipping'); raise SystemExit
text = path.read_text()
if 'mxfp8' in text.lower() and 'quant_key' in text:
    print('[fix-minimax-m3-sm121] MARLIN assertion already patched for MXFP8'); raise SystemExit
old = '''        assert (
            quant_config.use_mxfp4_w4a16
            or quant_config.use_nvfp4_w4a16
            or quant_config.use_int4_w4a16
            or quant_config.use_int8_w8a16
            or quant_config.use_fp8_w8a16
        ), "Supports only {mxfp,nvfp,int}4_w4a16, int8_w8a16 or fp8_w8a16"'''
new = '''        assert (
            quant_config.use_mxfp4_w4a16
            or quant_config.use_nvfp4_w4a16
            or quant_config.use_int4_w4a16
            or quant_config.use_int8_w8a16
            or quant_config.use_fp8_w8a16
            or quant_config.quant_key.value.startswith("f8e4m3fn,scale(u8,")
            or "mxfp8" in str(quant_config.quant_key)
        ), f"Unsupported MoE quant config: {quant_config.quant_key}"'''
if old in text:
    text = text.replace(old, new); path.write_text(text)
    print('[fix-minimax-m3-sm121] patched MARLIN MoE assertion for MXFP8')
else:
    print('[fix-minimax-m3-sm121] WARNING: MARLIN assertion pattern not found (skipping)')
PY

# ============================================================================
# 6. Patch api_server.py to skip startup reset_mm_cache
# ============================================================================
echo ""; echo "[fix-minimax-m3-sm121] === Step 6: Patch api_server.py ==="
python3 - <<'PY'
import pathlib, site
_site = [p for p in site.getsitepackages() if "dist-packages" in p or "site-packages" in p][0]
path = pathlib.Path(_site) / 'vllm/entrypoints/openai/api_server.py'
if not path.exists():
    print('[fix-minimax-m3-sm121] api_server.py not found; skipping'); raise SystemExit
text = path.read_text()
if 'reset_mm_cache' not in text:
    print('[fix-minimax-m3-sm121] reset_mm_cache not found; skipping'); raise SystemExit
if 'patched: skip on multi-node' in text:
    print('[fix-minimax-m3-sm121] api_server.py already patched')
else:
    text = text.replace('await async_llm.reset_mm_cache()',
        '# patched: skip on multi-node K2.6 first NCCL collective hangs\n        # await async_llm.reset_mm_cache()')
    path.write_text(text)
    print('[fix-minimax-m3-sm121] patched api_server.py -> skip startup reset_mm_cache')
PY

# ============================================================================
# 7. Fix MXFP4 MoE SwiGLU-OAI clamp (SWIGLUOAI_UNINTERLEAVE requires clamp_limit)
# ============================================================================
echo ""; echo "[fix-minimax-m3-sm121] === Step 7: Fix MXFP4 MoE SwiGLU-OAI clamp ==="
python3 - <<'PY'
import pathlib, site, re
_site = [p for p in site.getsitepackages() if "dist-packages" in p or "site-packages" in p][0]
path = pathlib.Path(_site) / ('vllm/model_executor/layers/quantization/compressed_tensors/'
                              'compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py')
if not path.exists():
    print('[fix-minimax-m3-sm121] compressed_tensors_moe_w4a4_mxfp4.py not found; skipping'); raise SystemExit
text = path.read_text()
if 'MINIMAX_M3_SWIGLU_CLAMP_FIX' in text:
    print('[fix-minimax-m3-sm121] MXFP4 SwiGLU clamp already patched'); raise SystemExit
pat = re.compile(r"    def get_fused_moe_quant_config\(\s*self.*?\n(?=    def )", re.DOTALL)
new = '''    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> "FusedMoEQuantConfig | None":
        # MINIMAX_M3_SWIGLU_CLAMP_FIX (0xSero/minimax-m3-sm120)
        if self.use_cutlass_mxfp4:
            cfg = mxfp4_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            )
        else:
            cfg = make_mxfp4_moe_quant_config(
                mxfp4_backend=self.mxfp4_backend,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                layer=layer,
            )
        if cfg is not None and getattr(self.moe, "swiglu_limit", None) is not None:
            cfg.gemm1_clamp_limit = self.moe.swiglu_limit
            cfg.gemm1_alpha = self.moe.swiglu_alpha
            cfg.gemm1_beta = self.moe.swiglu_beta
        return cfg

'''
text2, n = pat.subn(new, text, count=1)
if n != 1:
    print('[fix-minimax-m3-sm121] ERROR: get_fused_moe_quant_config not found'); raise SystemExit(1)
path.write_text(text2)
print('[fix-minimax-m3-sm121] patched MXFP4 MoE -> forwarded SwiGLU-OAI clamp')
PY

# ============================================================================
# 8. Verify
# ============================================================================
echo ""; echo "[fix-minimax-m3-sm121] === Verification ==="
python3 -c "
from pathlib import Path
import site
_site = [p for p in site.getsitepackages() if 'dist-packages' in p or 'site-packages' in p][0]
try:
    import fmha_sm12x; print('[OK] fmha_sm12x import')
except Exception as e: print(f'[FAIL] fmha_sm12x: {e}')
marlin = (Path(_site)/'vllm/model_executor/layers/fused_moe/experts/marlin_moe.py').read_text()
print('[OK] MARLIN MXFP8 assert' if 'mxfp8' in marlin else '[WARN] MARLIN assert not patched')
oracle = (Path(_site)/'vllm/model_executor/layers/fused_moe/oracle/mxfp8.py').read_text()
print('[OK] MXFP8 EMULATION' if 'EMULATION' in oracle else '[WARN] MXFP8 EMULATION not patched')
mxfp4 = (Path(_site)/'vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py').read_text()
print('[OK] MXFP4 SwiGLU clamp' if 'MINIMAX_M3_SWIGLU_CLAMP_FIX' in mxfp4 else '[FAIL] MXFP4 clamp NOT patched')
"
echo ""; echo "[fix-minimax-m3-sm121] All GB10 SM121 fixes complete"
