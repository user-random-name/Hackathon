#!/usr/bin/env python3
"""Convert a PEFT LoRA adapter trained on SmolLM2-360M-Instruct into a llama.cpp GGUF adapter.

This is a small, single-purpose converter (SHIPPED with the task). It writes the same GGUF
layout that llama.cpp's own convert_lora_to_gguf.py produces for a Llama-architecture
model: tensors "blk.N.<name>.weight.lora_a" / ".lora_b", plus the adapter metadata keys.
It supports adapters whose target modules are among q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj and down_proj (the standard choice). Embedding or lm_head adapters are
rejected on purpose.

Usage:
    python tools/lora_to_gguf.py <adapter_dir> --outfile adapter.gguf [--outtype f16|f32]

Requires: gguf==0.19.0, safetensors, numpy (no torch needed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

import gguf

N_BLOCKS = 32  # SmolLM2-360M has 32 transformer blocks
ARCH = gguf.MODEL_ARCH.LLAMA

# HF module name -> llama.cpp tensor name (per block)
MODULE_MAP = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def convert(adapter_dir: Path, outfile: Path, outtype: str = "f16") -> None:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    alpha = float(cfg["lora_alpha"])
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    writer = gguf.GGUFWriter(path=None, arch=gguf.MODEL_ARCH_NAMES[ARCH])
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, alpha)

    n_written = 0
    pairs: dict[str, dict[str, np.ndarray]] = {}
    for name, arr in tensors.items():
        # e.g. base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight
        if ".lora_A.weight" in name:
            kind = "a"
        elif ".lora_B.weight" in name:
            kind = "b"
        else:
            sys.exit(f"unsupported tensor in adapter: {name}")
        if "embed_tokens" in name or "lm_head" in name:
            sys.exit(f"embedding / lm_head adapters are not supported: {name}")
        try:
            after = name.split(".layers.", 1)[1]
            bid, rest = after.split(".", 1)
            module = rest.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
            dest = f"blk.{int(bid)}.{MODULE_MAP[module]}.weight"
        except (IndexError, KeyError, ValueError):
            sys.exit(f"cannot map tensor name: {name}")
        pairs.setdefault(dest, {})[kind] = arr

    dtype = np.float16 if outtype == "f16" else np.float32
    for dest in sorted(pairs, key=lambda d: (int(d.split(".")[1]), d)):
        ab = pairs[dest]
        if "a" not in ab or "b" not in ab:
            sys.exit(f"incomplete LoRA pair for {dest}")
        a = np.ascontiguousarray(ab["a"].astype(dtype))  # [r, in]
        b = np.ascontiguousarray(ab["b"].astype(dtype))  # [out, r]
        writer.add_tensor(dest + ".lora_a", a)
        writer.add_tensor(dest + ".lora_b", b)
        n_written += 2

    writer.write_header_to_file(path=str(outfile))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print(f"wrote {outfile} : {n_written} tensors, alpha={alpha}, dtype={outtype}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter_dir", type=Path, help="folder with adapter_config.json and adapter_model.safetensors")
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--outtype", choices=["f16", "f32"], default="f16")
    args = ap.parse_args(argv)
    convert(args.adapter_dir, args.outfile, args.outtype)


if __name__ == "__main__":
    main()
