#!/usr/bin/env python3
"""Patch sparse_attention.py to allow SM121 (GB10) for MSA sparse attention."""
import sys

path = None
import importlib.util
for p in [
    "/usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/common/sparse_attention.py",
]:
    try:
        with open(p) as f:
            if "is_device_capability_family(100)" in f.read():
                path = p
                break
    except FileNotFoundError:
        continue

if path is None:
    print("[patch_msa_sm121] sparse_attention.py not found; skipping")
    sys.exit(0)

with open(path, "r") as f:
    content = f.read()

old = 'and current_platform.is_device_capability_family(100)'
new = 'and (current_platform.is_device_capability_family(100) or current_platform.is_device_capability_family(120))'

if old not in content:
    print("[patch_msa_sm121] pattern already patched or not found; skipping")
    sys.exit(0)

content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print("[patch_msa_sm121] Patched sparse_attention.py to allow SM121 for MSA")
