#!/usr/bin/env bash
# NVFP4 weights + SageAttention on g7, at TP=4 x ULYSSES=2.
#
# WHY NVFP4 IS SECOND AND NOT FIRST. The g7e project's own ablation is that weight quantization
# alone is a REGRESSION at 768p -- it never touches attention, and attention is 59% of a 768p step,
# so 4-bit GEMMs shave the smaller half while the fp4 dequant sits in the way of the bigger half.
# It wins only stacked on sage. So sage was measured alone first (85.85 s t2va / 134.29 s ref2va,
# G7.md 2.1) and this script's question is narrow: does NVFP4 buy anything ON TOP of sage.
#
# NO SOURCE PATCHES. The g7e recipe carries two, and both are load-bearing there because both fail
# SILENTLY -- a wrong fp4 scale layout renders a grey smear, not an error. Neither is needed on this
# image (sglang 20518d85), and this was checked in the source rather than assumed:
#
#   patch_nvfp4_tma_scale_layout.py  -- NOT NEEDED, and its anchor is gone. That patch widened
#     `if flashinfer_backend is None or uses_flux1_scale_layout:` to also cover "auto"/"cutlass".
#     Upstream deleted the condition instead: modelopt_quant.py now early-returns for trtllm (:730)
#     and then applies the 128x4 TMA reshape/permute UNCONDITIONALLY, with the comment "Every FP4
#     GEMM reachable here reads block scales in the 128x4 TMA layout". H3_FP4_TMA_SCALES is
#     therefore a dead env var here; it is not passed.
#   patch_h3_qkv_scale_reorder.py    -- NOT NEEDED, and applying it would be ACTIVELY HARMFUL. The
#     fix landed generalised: minimax_h3.py now iterates `self.qkv_proj.named_parameters()` and
#     reorders every row-indexed sibling of the permuted qkv weight, gating on row count, and it
#     handles BlockQuantScaleParameter by scaling both the permutation and the gate down by the
#     block height -- where the patch only knew the two names weight_scale/weight_scale_inv. Adding
#     the patch on top would permute the rows TWICE, which is exactly as wrong as not permuting them
#     at all, and just as silent.
#
# DO NOT ADD --quantization modelopt_fp4. The checkpoint carries its own `_quantization_metadata`
# header (written by nvfp4_quantize_transformer.py) and quantization_utils.py:131 reads the layer
# markers straight out of it; passing the flag builds a second, empty config that wins and serves an
# unquantized-but-declared-quantized model. QUANT= (empty) is what keeps --quantization off the
# command line entirely -- sglang_base_arm.sh:89 only adds the flag when QUANT is non-empty.
#
# WHAT IS UNVALIDATED HERE, AND IT IS THE ONE REAL RISK. g7e never ran NVFP4 at TP>1: every arm of
# theirs is GPUS=1 ULYSSES=1 or GPUS=2 ULYSSES=2 on a 96 GB card. Our 32 GB cards force TP>=2, and
# the qkv scale reorder runs in the weight loader where a rank sees only its own shard. Upstream's
# version sets rank_local_weight_transform for the scales as well as the weights, which is the hook
# that exists precisely for this, so it should hold -- but "should" is why THE FIRST RENDER OF EACH
# ARM MUST BE WATCHED. A wrong per-row scale under TP produces a flat grey blur and no error, and it
# would otherwise be reported here as a successful 60 s render.
#
# HELD FIXED against the sage arms: base model, 768p, 121 f (5.04 s), seed 42, 25 steps, TP=4 x
# ULYSSES=2, reference short edge 2048, sage on every arm. The only change is where the transformer
# weights come from. t2va uses case_t2va_v2.txt @wide (see sage.sh's header for why, not case_ir.txt).
set -uo pipefail
V=/data/h3; L=$V/sglang/logs; R=$L/nvfp4.txt
export ROOT=$V/sglang VDNROOT=$V FRAMES=121 OUTDIR=$V/pull/case REFDIR=$V/ref
: > $R
SAGE=(--attention-backend sage_attn
      --component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa)
# RUNAI streamer off: it streams shards straight to device and does not carry the safetensors
# metadata header through, which is the only thing telling the loader these tensors are nvfp4.
# FP4 GEMM backend "auto": flashinfer has no trtllm fp4 GEMM at capability 120
# ("mm_fp4 does not support backend 'trtllm' with capability 120"), so leaving this unset picks a
# backend that does not exist on this card.
export SGLANG_USE_RUNAI_MODEL_STREAMER=0
export SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=auto

up() { local log=$1 i; for i in $(seq 1 40); do
  grep -q "fired up and ready to roll" "$log" 2>/dev/null && { echo "  ready ${i}0s" | tee -a $R; return 0; }
  grep -q "Error while loading component\|Server warmup failed\|processing failed" "$log" 2>/dev/null && { echo "  FAILED" | tee -a $R; grep -m3 -i "outofmemory\|Error" "$log" | cut -c1-260 | tee -a $R; return 1; }
  pgrep -f '[s]glang.*serve' >/dev/null || { echo "  SERVER GONE" | tee -a $R; tail -6 "$log" | cut -c1-260 | tee -a $R; return 1; }
  sleep 10; done; echo "  TIMEOUT" | tee -a $R; return 1; }
stop() { pkill -f '[s]glang.*serve' >/dev/null 2>&1; sleep 12; }
want() { case " ${ARMS:-N1 N2} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Both readbacks come out of the server's own log, because both failure modes are silent and both
# reproduce a number we already have: no sage falls back to torch_sdpa (= the 105/177 s baseline),
# and no fp4 serves bf16 weights (which on 32 GB cards is more likely to OOM than to be quiet, but
# the check is free). The fp4 line is only reported, not asserted, since the exact wording of the
# quantization log varies; the render is skipped only if sage is missing.
readback() { local log=$1 line
  line=$(grep -h "Using .* attention backend" "$log" | tail -1)
  echo "  attention: ${line:-<none found>}" | tee -a $R
  grep -hi "nvfp4\|modelopt\|fp4" "$log" | grep -vi "does not support" | tail -4 \
    | cut -c1-200 | sed 's/^/  quant: /' | tee -a $R
  case "$line" in
    *sage_attn*) return 0 ;;
    *) echo "  !! SAGE NOT ACTIVE -- skipping the render." | tee -a $R; return 1 ;;
  esac; }

stop
if want N1; then
echo "=== N1  ref2va + nvfp4 + sage   (vs sage 134.29 s, fp8+sdpa 177.2 s)" | tee -a $R
log=$L/serve_ref2va_768p_nvfp4.log; rm -f $log
QUANT= GPUS=8 TP=4 ULYSSES=2 LOGTAG=nvfp4 \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
    --transformer-weights-path $V/nvfp4_ref2va.safetensors "${SAGE[@]}" \
    > $L/launch_nvfp4_a.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=nvfp4 768:25:121 2>&1 | tee -a $R
fi
stop
fi

if want N2; then
echo "=== N2  t2va@wide + nvfp4 + sage  (vs sage 85.64 s, fp8+sdpa 105.4 s)" | tee -a $R
log=$L/serve_base_768p_nvfp4.log; rm -f $log
# nvfp4_fl2va, not a "t2va" file: MINIMAX_H3_TASK_PARTITIONS maps t2va -> fl2va, so the base server
# holds the FL2VA weight partition and that is the transformer being replaced.
QUANT= GPUS=8 TP=4 ULYSSES=2 LOGTAG=nvfp4 \
  setsid nohup bash $V/sglang_base_arm.sh serve 768 \
    --transformer-weights-path $V/nvfp4_fl2va.safetensors "${SAGE[@]}" \
    > $L/launch_nvfp4_b.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  python $V/sglang_case.py case=$V/case_t2va_v2.txt task=t2va label=wide tag=nvfp4 768:25:121 2>&1 | tee -a $R
fi
stop
fi
echo NVFP4_DONE | tee -a $R
