#!/usr/bin/env python3
"""Offline per-tensor FP8 (e4m3) export of one H3 DiT partition.

    SRC=<snapshot>/Ref2VA/transformer DST=/data/h3/fp8_ref2va.safetensors python3 fp8_quantize_transformer.py

WHY THIS EXISTS. `--quantization fp8` is *online*: the loader lands the 65.65 GiB bf16 checkpoint on
the cards and casts there, which is 32.8 GiB/card at TP=2 against 31.37 GiB of card. Arm T4 proved
`--dit-layerwise-offload` does not rescue it -- with both flags active the process still dies at
29.47 GiB in fsdp_load.py:load_model_from_full_model_state_dict, because the quantized load path
materialises parameters on device to attach weight scales. A pre-quantized file needs no cast at all,
so it is the only route to fp8 below TP=4. See G7.md 3.1.1.

FORMAT: Comfy per-layer fp8, which is what this image's loader calls
`quantization_utils.resolve_comfy_checkpoint_quantization` -> `ComfyFp8Config` when the format SET of
the `_quantization_metadata` markers is exactly ["float8_e4m3fn"] (:407). Each marked layer then gets
the ordinary `Fp8LinearMethod` with `is_checkpoint_fp8_serialized=True`, and the activation scheme is
inferred per layer from the checkpoint: `static` if a `<prefix>.input_scale` exists, else `dynamic`
(quantization_utils.py:230-233). We write no input_scale, so every layer is dynamic -- the same
activation handling online fp8 uses, which is what makes F1 below a fair control.

*** THE ONE THING THAT WOULD FAIL SILENTLY, AND WHY THE QKV REORDER IS NOT OPTIONAL ***
`ComfyFp8Config.checkpoint_uses_native_qkv_layout = True` (comfy_fp8.py:91). minimax_h3.py:798-806
reads exactly that attribute, and when it is true it SKIPS `_install_qkv_weight_loader` entirely. So
declaring a layer fp8 is also asserting that its qkv rows are already in SGLang's native
[q_all, k_all, v_all] order -- but the official H3 safetensors interleave Q/K/V per head. Writing a
straight cast would hand the model a per-head-interleaved tensor while telling it the tensor is
native, and it would render garbage with no error, exactly like the NVFP4 arms did (TODO.md).
This is why NVFP4 got away without it: ComfyNvfp4Config leaves the attribute False, so the loader's
own reorder stays installed. FP8 flips the contract.

We therefore reorder here, and we do it by importing SGLang's own
`_reorder_grouped_qkv_to_qkv` rather than reimplementing it, so the two cannot drift. The call is
minimax_h3.py:884-889's: num_query_groups = num_attention_heads, heads_per_group = 1 (H3 is MHA),
head_dim = attention_head_dim.

SCALE SHAPE IS PER LOGICAL MATRIX, NOT PER TENSOR. Fp8LinearMethod registers
`PerTensorScaleParameter(data=torch.empty(len(output_partition_sizes)))` (fp8.py:220), i.e. one fp32
scale per *logical* matrix of a fused linear: 3 for qkv_proj, 2 for fc1 ([gate, up]), 1 for the two
RowParallelLinears. `process_weights_after_loading` then calls `requantize_with_max_scale` to bring
the slices onto a common scale, so per-slice quantization here is what that function expects to find.
A single scalar would be the wrong shape and is rejected rather than broadcast.

ADALN STAYS BF16, same as the NVFP4 export, and for a better reason than there. minimax_h3.py:2081
materialises those weights in fp32 whenever curve AdaLN is on, so quantizing them buys file size and
nothing else -- they are ~40% of the DiT's weights at 0% of its FLOPs. Leaving them alone also keeps
the quantized layer set byte-for-byte the same 208 as the NVFP4 export, which is the set already
proven to be constructed with a quant_config (a layer built with `quant_config=None`, e.g.
`to_gate_compress`, would keep a bf16 parameter and fail the load on dtype).

EXPECTED SELF-CHECK. Per-tensor e4m3 has 3 mantissa bits, so the round-trip relative error floor is
~0.02-0.04 -- about a third of NVFP4's 0.094 group-16 floor. The script refuses to write if it
quantized anything other than 208 layers or if the worst error exceeds 0.08, which is the difference
between finding out here and finding out from a broken video twenty minutes later.

FILE SIZE, so nobody is surprised: ~45 GiB (the 208 linears are 41 GiB bf16 -> 20.5 GiB fp8, and the
24.47 GiB of adaln and friends are copied through). That is 11.2 GiB/card at TP=4 and 22.5 GiB/card
at TP=2 -- the second is the whole point, and it is why the TP=2 arm needs no weight streaming.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The SAME layer set as the NVFP4 export, imported rather than copied: 52 blocks (50 + 2 token
# refiner) x {attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2} = 208.
from nvfp4_quantize_transformer import QUANT_RE  # noqa: E402
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (  # noqa: E402
    _reorder_grouped_qkv_to_qkv,
)

FP8_MAX = 448.0
# One fp32 scale per logical matrix; see the docstring. Keyed by the suffix before `.weight`.
LOGICAL_WIDTHS = {"attn.qkv_proj": 3, "mlp.fc1": 2}


def logical_slices(name: str) -> int:
    for suffix, n in LOGICAL_WIDTHS.items():
        if name.endswith(f"{suffix}.weight"):
            return n
    return 1


def quantize_fp8(w: torch.Tensor, n_slices: int):
    """bf16 [N, K] -> (F8_E4M3 [N, K], F32 [n_slices]), quantized per logical output slice."""
    n, k = w.shape
    assert n % n_slices == 0, (n, n_slices)
    rows = n // n_slices
    q = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    scales = torch.empty(n_slices, dtype=torch.float32)
    for i in range(n_slices):
        sl = w[i * rows : (i + 1) * rows].float()
        # amax/FP8_MAX puts the largest magnitude of the slice exactly at e4m3's top finite value.
        s = (sl.abs().amax() / FP8_MAX).clamp(min=1e-30)
        scales[i] = s
        q[i * rows : (i + 1) * rows] = (sl / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scales


def dequant_fp8(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows = q.shape[0] // scales.numel()
    out = q.float()
    for i in range(scales.numel()):
        out[i * rows : (i + 1) * rows] *= scales[i]
    return out


def main() -> int:
    src = os.environ["SRC"]
    dst = os.environ["DST"]
    heads = int(os.environ.get("HEADS", "56"))          # num_attention_heads
    head_dim = int(os.environ.get("HEAD_DIM", "128"))   # attention_head_dim
    check_every = int(os.environ.get("CHECK_EVERY", "40"))

    index = os.path.join(src, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))

    out, quant_meta = {}, {}
    n_q = n_copy = n_reorder = 0
    worst = (0.0, "")
    for shard in shards:
        with safe_open(os.path.join(src, shard), "pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if not QUANT_RE.search(name):
                    out[name] = t
                    n_copy += 1
                    continue
                if name.endswith("attn.qkv_proj.weight"):
                    # Per-head [q, k, v] -> [q_all, k_all, v_all]. MUST happen before the slice
                    # split below, because the three logical matrices only exist after it.
                    t = _reorder_grouped_qkv_to_qkv(
                        t, num_query_groups=heads, heads_per_group=1, head_dim=head_dim
                    )
                    n_reorder += 1
                n_slices = logical_slices(name)
                q, scales = quantize_fp8(t, n_slices)
                out[name] = q
                out[name + "_scale"] = scales
                quant_meta[name[: -len(".weight")]] = {"format": "float8_e4m3fn"}
                n_q += 1
                if n_q % check_every == 1:
                    ref = t.float()
                    rel = float((dequant_fp8(q, scales) - ref).norm() / ref.norm())
                    print("  check %-46s slices=%d rel=%.4f" % (name, n_slices, rel), flush=True)
                    if rel > worst[0]:
                        worst = (rel, name)
        print("shard %s done (q=%d copy=%d)" % (shard, n_q, n_copy), flush=True)

    if n_q != 208:
        print("!!! quantized %d linears, expected 208 -- QUANT_RE does not match this "
              "checkpoint" % n_q)
        return 1
    if n_reorder != 52:
        print("!!! reordered %d qkv tensors, expected 52 -- the qkv naming changed, and a "
              "MISSED reorder renders garbage SILENTLY" % n_reorder)
        return 1
    if worst[0] > 0.08:
        print("!!! round-trip rel err %.4f (%s) is far above the per-tensor e4m3 floor" % worst)
        return 1

    save_file(
        out,
        dst,
        metadata={
            "_quantization_metadata": json.dumps(
                {"format_version": "1.0", "layers": quant_meta}
            ),
            "target_format": "FP8_E4M3",
            "converted_by": "fp8_quantize_transformer.py",
        },
    )
    print("wrote %s: %d tensors (q=%d copy=%d qkv_reordered=%d) worst rel=%.4f (%s)"
          % (dst, len(out), n_q, n_copy, n_reorder, worst[0], worst[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
