#!/usr/bin/env python3
"""Rewrite our NVFP4 H3 checkpoint into the layout this image's H3 loader *asserts* it is in.

    CHECK=1 SRC=/data/h3/nvfp4_ref2va.safetensors BF16=<snapshot>/Ref2VA/transformer \
      python3 /data/h3/nvfp4_comfy_layout.py          # reads 3 layers, writes nothing
    SRC=... DST=/data/h3/nvfp4c_ref2va.safetensors BF16=... python3 /data/h3/nvfp4_comfy_layout.py

WHY THIS EXISTS: it is the fix for the NVFP4 noise, and the noise was never about tensor parallelism.

`--transformer-weights-path` on H3 goes through `resolve_minimax_h3_checkpoint_quantization`
(`runtime/loader/minimax_h3_weights.py:61`, called from
`runtime/loader/component_loaders/transformer_loader.py:315`). If ANY layer marker says `nvfp4`, that
function does not infer the layout -- it *hard-codes* three properties of the file (`:85-88`):

    config.checkpoint_uses_native_qkv_layout = True     # so minimax_h3.py:800 installs NO qkv reorder
    config.checkpoint_weight_scale_layout = "swizzled"   # so the loader UNswizzles weight_scale
    config.swap_weight_nibbles = True                    # so the loader SWAPS every nibble

and the two consumers are `modelopt_quant.py:718` (`_swizzled_nvfp4_scales_to_linear`) and `:707`
(`_prepare_nvfp4_weight_bytes`). That is the ComfyUI-kitchen convention, and our file is the exact
opposite on all three axes, because `nvfp4_quantize_transformer.py` (g7e's, copied verbatim) writes
the *canonical* convention and documents it in its own header: linear scales, low nibble = even
index, and "qkv 的行序:**不要动**" because the loader used to do that reorder. On this image the
loader does not, so:

    * 52 qkv_proj layers keep per-head [q,k,v] rows while the model reads [q_all,k_all,v_all]
    * all 208 weight_scale tensors get un-swizzled, i.e. scrambled, because they were never swizzled
    * all 208 weights get their nibbles swapped, i.e. every value is replaced by its neighbour

Any ONE of those is fatal on its own; g7e measured cos ~0.000 / rel ~1.41 for a single wrong axis
against cos 0.9955 / rel 0.094 with all three right (`nvfp4_canonicalize.py`'s header). Three at once
is why both NVFP4 arms rendered 12.5 Mbps noise with no error in the log, at TP=4 -- and it would have
rendered noise at TP=1 too. G7.md 3.2.2's tensor-parallel hypothesis is dead, and TODO.md's "cleared
suspect" note on `checkpoint_uses_native_qkv_layout` was reading the class default
(`configs/base_config.py:38`) instead of the H3 override above.

THE METHOD IS THE FP8 ONE. `fp8_quantize_transformer.py` had the same problem in the other precision
-- `ComfyFp8Config.checkpoint_uses_native_qkv_layout = True` -- and solved it by doing the reorder in
the exporter with the model's own `_reorder_grouped_qkv_to_qkv`. Same here, plus two layout fixes fp8
does not need. Every one of the three transforms is a **pure permutation or a nibble exchange**, so
this adds no quantization error at all: the round-trip error of the output file is exactly the input
file's 0.094, which is the group-16 e2m1 floor.

WHY CONVERT RATHER THAN RE-QUANTIZE. Nothing about the arithmetic was wrong, only the layout, so
re-running the quantizer would reproduce the same numbers and the same mismatch. Converting also
keeps g7e's exporter byte-for-byte theirs (see quant.sh) and is one 37.5 GB read instead of a 65 GB
one.

THE CHECK IS THE POINT. `CHECK=1` dequantizes a few layers **the way this image's loader will**
(`_prepare_nvfp4_weight_bytes` -> `_swizzled_nvfp4_scales_to_linear` -> `srt`'s `dequantize_nvfp4`,
all imported, none re-derived) and compares against the original bf16 weight, before and after the
transform. Expect roughly:

    blocks.0.attn.qkv_proj    now cos=0.000 rel=1.41    fixed cos=0.995 rel=0.094
    blocks.0.mlp.fc1          now cos=0.000 rel=1.41    fixed cos=0.995 rel=0.094

`now cos` near zero is the diagnosis; `fixed cos` near 0.995 is the fix; both are free of GPU time.
A conversion that got the qkv rows wrong is invisible in the fc1 row and visible in the qkv row --
that is why the check picks one of each, and why the write refuses unless it saw 52 qkv reorders and
208 quantized layers. It is the same refusal-gate discipline as the fp8 exporter, for the same reason:
this failure mode renders cleanly.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (  # noqa: E402
    _prepare_nvfp4_weight_bytes,
    _swizzled_nvfp4_scales_to_linear,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (  # noqa: E402
    _reorder_grouped_qkv_to_qkv,
)
from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4  # noqa: E402


def swizzle(scale: torch.Tensor) -> torch.Tensor:
    """Linear (row-indexed) fp8 block scales -> the 128x4-tile layout the loader will un-swizzle.

    Exact inverse of modelopt_quant.py's `_swizzled_nvfp4_scales_to_linear`, which reads
    (M/128, K/4, 32, 4, 4) and permutes (0,1,4,3,2,5). That permutation swaps axes 2 and 4 and is
    therefore its own inverse, so only the reshape differs. The final reshape is a flat
    reinterpretation, not a dim-product match -- which is also how the forward function works.
    Verified end to end by the cos check rather than by this comment.
    """
    m, k = scale.shape
    if m % 128 or k % 4:
        raise ValueError(f"scale {(m, k)} is not 128x4-tile aligned; needs a padding path")
    return (
        scale.reshape(m // 128, 4, 32, k // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(m, k)
    )


def swap_nibbles(weight: torch.Tensor) -> torch.Tensor:
    """Low-nibble-even -> high-nibble-even. The loader swaps back (swap_weight_nibbles=True)."""
    return ((weight & 0x0F) << 4) | ((weight >> 4) & 0x0F)


def runtime_view(weight, scale, scale_2):
    """Dequantize exactly as this image's loader will read the file. No convention assumed here."""
    return dequantize_nvfp4(
        _prepare_nvfp4_weight_bytes(weight, swap_weight_nibbles=True),
        _swizzled_nvfp4_scales_to_linear(scale),
        scale_2,
        out_dtype=torch.float32,
    )


def agreement(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    # float64, not float32: these tensors are 29-87 M elements and an fp32 dot product accumulates
    # enough error to print a cosine of 1.036, which is not a number a cosine can take and reads as a
    # broken check rather than a passing one.
    a, b = got.flatten().double(), ref.flatten().double()
    return (
        float(F.cosine_similarity(a, b, dim=0)),
        float((a - b).norm() / b.norm()),
    )


class Bf16Reference:
    """Lazy per-key reader over the original bf16 shards, for the checks only."""

    def __init__(self, root: str):
        self.root = root
        index = os.path.join(root, "model.safetensors.index.json")
        with open(index) as f:
            self.weight_map = json.load(f)["weight_map"]

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        with safe_open(os.path.join(self.root, shard), "pt") as f:
            return f.get_tensor(name)


def main() -> int:
    src = os.environ["SRC"]
    dst = os.environ.get("DST")
    bf16 = os.environ.get("BF16")
    heads = int(os.environ.get("HEADS", "56"))
    head_dim = int(os.environ.get("HEAD_DIM", "128"))
    check_every = int(os.environ.get("CHECK_EVERY", "40"))
    check_only = os.environ.get("CHECK") == "1"
    if not check_only and not dst:
        print("DST is required unless CHECK=1")
        return 1
    ref = Bf16Reference(bf16) if bf16 else None

    def reorder(t):
        return _reorder_grouped_qkv_to_qkv(
            t, num_query_groups=heads, heads_per_group=1, head_dim=head_dim
        )

    def check(mod, weight, scale, scale_2, is_qkv):
        """Print, for one module, what the loader sees now and what it would see after the fix."""
        target = ref.get(f"{mod}.weight").float()
        if is_qkv:
            target = reorder(target)
        now = agreement(runtime_view(weight, scale, scale_2), target)
        fixed = agreement(
            runtime_view(
                swap_nibbles(reorder(weight) if is_qkv else weight),
                swizzle(reorder(scale) if is_qkv else scale),
                scale_2,
            ),
            target,
        )
        print(
            "  %-46s now cos=%.3f rel=%.3f   fixed cos=%.3f rel=%.3f"
            % (mod, now[0], now[1], fixed[0], fixed[1]),
            flush=True,
        )
        return fixed

    out = {}
    n_q = n_qkv = n_copy = 0
    worst = (1.0, 0.0, "")  # (min cos, max rel, name)
    with safe_open(src, "pt") as f:
        keys = list(f.keys())
        metadata = f.metadata() or {}
        quant_mods = {k[: -len(".weight_scale")] for k in keys if k.endswith(".weight_scale")}
        print(f"{len(keys)} tensors, {len(quant_mods)} quantized modules", flush=True)
        for key in sorted(keys):
            mod, _, leaf = key.rpartition(".")
            if mod not in quant_mods or leaf not in ("weight", "weight_scale"):
                if not check_only:
                    out[key] = f.get_tensor(key)
                n_copy += 1
                continue
            if leaf == "weight_scale":
                continue  # handled with its weight, below
            is_qkv = mod.endswith("qkv_proj")
            weight = f.get_tensor(key)
            scale = f.get_tensor(f"{mod}.weight_scale")
            scale_2 = f.get_tensor(f"{mod}.weight_scale_2")
            n_q += 1
            n_qkv += is_qkv
            # One of each kind, early, is what makes CHECK=1 cheap and still conclusive: a wrong qkv
            # permutation shows up only in a qkv row, a wrong swizzle/nibble in either.
            do_check = ref is not None and (
                (check_only and (is_qkv or n_q <= 2)) or (not check_only and n_q % check_every == 1)
            )
            if do_check:
                cos, rel = check(mod, weight, scale, scale_2, is_qkv)
                worst = (min(worst[0], cos), max(worst[1], rel), mod)
            if check_only:
                if n_q >= int(os.environ.get("LAYERS", "3")) and n_qkv:
                    break
                continue
            if is_qkv:
                weight, scale = reorder(weight), reorder(scale)
            out[key] = swap_nibbles(weight).contiguous()
            out[f"{mod}.weight_scale"] = swizzle(scale)
            if n_q % 50 == 0:
                print(f"  {n_q}/208 quantized, {n_copy} copied", flush=True)

    if check_only:
        print("CHECK only, nothing written. worst fixed cos=%.3f rel=%.3f" % worst[:2])
        return 0

    # Same gates as the fp8 exporter, and for the same reason: every one of these failures produces a
    # file that loads and renders. 52 = 50 blocks + 2 token_refiner blocks.
    if n_q != 208:
        print(f"!!! {n_q} quantized layers, expected 208 -- wrong file or wrong key names")
        return 1
    if n_qkv != 52:
        print(f"!!! reordered {n_qkv} qkv tensors, expected 52 -- a MISSED reorder renders garbage")
        return 1
    if ref is not None and (worst[0] < 0.99 or worst[1] > 0.12):
        print("!!! worst checked layer cos=%.3f rel=%.3f (%s) -- transform is wrong" % worst)
        return 1
    if ref is None:
        print("!!! refusing to write unverified: set BF16=<snapshot>/<V>/transformer")
        return 1

    print(f"writing {dst}", flush=True)
    save_file(out, dst, metadata=metadata or None)
    print(
        "wrote %s: %d tensors (q=%d qkv_reordered=%d copy=%d) worst cos=%.3f rel=%.3f"
        % (dst, len(out), n_q, n_qkv, n_copy, worst[0], worst[1])
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
