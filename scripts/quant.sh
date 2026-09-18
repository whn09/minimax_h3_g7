#!/usr/bin/env bash
# Offline NVFP4 conversion of the two H3 DiT partitions, run inside the h3-serve pod.
#
#   kubectl exec h3-serve -- bash -lc 'setsid nohup bash /data/h3/quant.sh > /data/h3/logs/quant.txt 2>&1 &'
#
# WHY OFFLINE AT ALL. `--quantization` does ONLINE quantization for fp8 and mxfp4 only; there is no
# online nvfp4 path, so the 4-bit weights have to exist as a file before the server starts. That is
# also what makes NVFP4 loadable on 32 GB cards where online fp8 is not: online fp8 lands the
# 65.65 GiB bf16 checkpoint on the cards *first* and casts there, whereas a pre-quantized file lands
# 36 GB, i.e. 9 GB per card at TP=4.
#
# THE QUANTIZER IS NOT OURS -- it is nvfp4_quantize_transformer.py from the sibling g7e project
# (Trn2/minimax_h3_g7e/scripts/). It is copied to the pod as-is and this file only drives it, so
# there is one implementation of the recipe and no chance of the two drifting. Recipe: 208 linear
# layers (52 x qkv/out/fc1/fc2) -> nvfp4, i.e. e2m1 nibbles + an fp8 per-16 block scale +
# a per-tensor fp32 scale; everything else copied through as bf16. 951 tensors out.
#
# ONE DELIBERATE DEPARTURE FROM THE g7e RECIPE, AND IT IS FORCED. Theirs also casts the 50
# `blocks.*.adaln_proj.linear.weight` tensors to bare fp8 (no scale) -- 40% of the DiT's weights at
# 0% of its FLOPs, so purely a file-size trick. This image REFUSES to load that, in two independent
# places, and it refuses loudly rather than rendering something wrong:
#
#   quantization_utils.py:160  any `.weight` stored as F8_E4M3/I8/U8 is treated as quantized and
#     must carry a comfy_quant marker, so the undeclared adaln casts fail as
#     "Quantized weights are missing comfy_quant metadata: ['blocks.0.adaln_proj.linear', ...]".
#   quantization_utils.py:206  declaring them `float8_e4m3fn` does not help twice over: that format
#     *requires* a `<prefix>.weight_scale` tensor, which a bare cast does not have, and :409 accepts
#     only ["nvfp4"] or ["int8_tensorwise","nvfp4"] as a format SET -- a float8+nvfp4 mix raises
#     "Unsupported Comfy quantization format(s)".
#
# So adaln stays bf16 here. That is free in speed (0% of the FLOPs), costs 24.47 -> 36 GB on disk and
# 6.1 -> 9.0 GB per card at TP=4, and is the *less* lossy of the two options. It is switched off by
# monkeypatching FP8_RE to a never-matching pattern rather than by editing their file, so the copy on
# the pod stays byte-identical to the g7e original.
#
# READ THE SELF-CHECK LINE, DO NOT SKIP IT. Expected, per partition:
#   wrote /data/h3/nvfp4_<v>.safetensors: 951 tensors (q=208 fp8=0 copy=327) worst rel=0.09xx
# 0.094±0.002 is the group-16 round-to-nearest-e2m1 floor. Materially higher means the packing or
# the scales are wrong; materially lower is impossible and means the check compared against the
# wrong tensor. The script exits non-zero if it quantized anything other than 208 layers or if the
# worst rel error exceeds 0.12, which is the difference between finding out here and finding out
# from a grey video twenty minutes later.
#
# SEQUENTIAL, AND NOT ALONGSIDE A TIMED REQUEST. It is pure CPU (~2 min per partition on this box,
# the g7e note says ~10 on theirs) with peak host RAM about one output file -- against 726 GB here,
# so RAM is not the constraint. The constraint is that a 32-thread torch job on the same host skews
# any inference number measured next to it.
set -uo pipefail
S=/data/h3/hf/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
cd /data/h3
for v in Ref2VA FL2VA; do
  lv=$(echo "$v" | tr A-Z a-z)
  echo "=== $v -> /data/h3/nvfp4_$lv.safetensors  $(date +%T)"
  SRC=$S/$v/transformer DST=/data/h3/nvfp4_$lv.safetensors \
  python3 - <<'PY'
import re, sys
sys.path.insert(0, "/data/h3")
import nvfp4_quantize_transformer as q
# `(?!)` is a zero-width negative lookahead on the empty string: it can never match, so every adaln
# weight falls through to the plain bf16 copy branch. See the header for why.
q.FP8_RE = re.compile(r"(?!)")
sys.exit(q.main())
PY
  echo "exit $?  $(date +%T)"; ls -l "/data/h3/nvfp4_$lv.safetensors" 2>/dev/null
done
echo QUANT_DONE
