#!/usr/bin/env bash
# What the three CPU-offload flags actually cost, at the measured floor.
#
#   kubectl exec h3-serve -- bash -lc 'cd /data/h3 && ARMS="A0 A1 A2 A3" setsid nohup bash offload.sh \
#     > /data/h3/sglang/logs/offload_run.log 2>&1 &'
#
# WHY. The floor (N8: NVFP4 + TP=1 x U=8 + sage + Cache-DiT rdt 0.16, 48.46 s) spends 41.07 s of its
# 48.46 in MiniMaxH3DenoisingStage and **6.2 s in stages that Cache-DiT cannot touch**:
# TextEncoding 1.75, VisualEncoding 0.60, MiniMaxH3DecodingStage 3.85. Every one of those three is
# carrying a CPU offload, and at TP=1 the card is only 11 788 of 31 373 MB full -- ~19.6 GB idle. So
# this measures a purely LOSSLESS saving: no new approximation, no quality question, just weights
# that no longer have to cross PCIe once per request. If it buys 3-4 s it is worth more than another
# turn of the Cache-DiT dial, which buys speed by skipping computation.
#
# WHY IT IS AN ABLATION AND NOT ONE ARM. Dropping all three at once is the arm most likely to OOM
# (text_encoder alone is ~7.2 GB -- the 28 142 vs 20 968 MB gap between N4 and the text-offloaded
# noise arm), and an OOM would leave us knowing nothing. Staged from cheapest-to-hold to most
# expensive, the run still answers the question even if the last arm dies:
#
#   A0  all three offloads                     control, expect 48.46 s / 11 788 MB
#   A1  - vae-cpu-offload                       targets Decoding 3.85 s; video_vae is 5.2 GB fp16
#   A2  - vae - image-encoder                   targets VisualEncoding 0.60 s (ref2va's ref tower)
#   A3  - vae - image-encoder - text_encoder    targets TextEncoding 1.75 s; ~+7.2 GB, the risky one
#
# --dit-layerwise-offload IS NOT ABLATED. At TP=1 it is not a speed knob, it is the only reason the
# DiT fits at all: 37.5 GB of weights, 32 GB cards, and Ulysses shards activations rather than
# weights. Removing it is a load OOM by construction. Its own cost is already bounded by N5 vs N6
# (37.5 GB/card/step streamed beat 18.7 GB/card/step by 5 %), i.e. hidden behind compute.
#
# ref2va ONLY, on purpose. It is the task under study, and it is also the only one of the two that
# exercises the image encoder at all -- t2va has no reference tower, so A2 would be a no-op there.
set -uo pipefail
V=/data/h3; L=$V/sglang/logs; R=$L/offload.txt
export ROOT=$V/sglang VDNROOT=$V FRAMES=121 OUTDIR=$V/pull/case REFDIR=$V/ref
F=$V/nvfp4c_ref2va.safetensors
mkdir -p "$L"
echo "########## $(date +%F' '%T)  ARMS=${ARMS:-<default>}" >> $R
SAGE=(--attention-backend sage_attn
      --component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa)
# Same two as nvfp4c.sh: the RUNAI streamer drops the safetensors metadata header that is the only
# thing marking these tensors nvfp4, and QUANT= (empty) keeps --quantization off the command line so
# the file's own header is the single source of the precision.
export SGLANG_USE_RUNAI_MODEL_STREAMER=0
export SGLANG_CACHE_DIT_ENABLED=true

up() { local log=$1 i; for i in $(seq 1 40); do
  grep -q "fired up and ready to roll" "$log" 2>/dev/null && { echo "  ready ${i}0s" | tee -a $R; return 0; }
  grep -q "Error while loading component\|Server warmup failed\|processing failed" "$log" 2>/dev/null && { echo "  FAILED" | tee -a $R; grep -m3 -i "outofmemory\|Error\|ValueError\|RuntimeError" "$log" | cut -c1-260 | tee -a $R; return 1; }
  pgrep -f '[s]glang.*serve' >/dev/null || { echo "  SERVER GONE" | tee -a $R; tail -6 "$log" | cut -c1-260 | tee -a $R; return 1; }
  sleep 10; done; echo "  TIMEOUT" | tee -a $R; return 1; }
stop() { pkill -f '[s]glang.*serve' >/dev/null 2>&1; sleep 12; }
want() { case " ${ARMS:-A0 A1 A2 A3} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Read the offload flags back out of the server's OWN parsed args rather than trusting the command
# line. This is the whole experiment's independent variable, and a silently-ignored flag would show
# up as "offload costs nothing", which is exactly the wrong conclusion to draw for free.
readback() { local log=$1 line
  line=$(grep -h "Using .* attention backend" "$log" | tail -1)
  echo "  attention: ${line:-<none found>}" | tee -a $R
  grep -m1 "server_args:" "$log" | tr ',' '\n' | grep -i offload | sed 's/^ */  flag: /' | tee -a $R
  case "$line" in
    *sage_attn*) return 0 ;;
    *) echo "  !! SAGE NOT ACTIVE -- skipping the render." | tee -a $R; return 1 ;;
  esac; }

# The per-stage breakdown is the point: the totals cannot say WHICH stage got cheaper, and the three
# flags target three different stages. Nine stages per request, so the last nine lines are the render
# we just did (warmup's breakdown is suppressed by RequestMetrics.suppress_stage_breakdown).
stages() { grep -h "\] finished in" "$1" | tail -9 \
  | sed 's/^\[[^]]*\] //; s/^/  stage: /' | tee -a $R; }

arm() { local name=$1; shift
  echo "=== $name  extra flags: ${*:-<none beyond --dit-layerwise-offload>}" | tee -a $R
  local log=$L/serve_ref2va_768p_ab$name.log; rm -f $log
  QUANT= GPUS=8 TP=1 ULYSSES=8 LOGTAG=ab$name \
    setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
      --transformer-weights-path $F --dit-layerwise-offload "$@" "${SAGE[@]}" \
      > $L/launch_ab$name.log 2>&1 < /dev/null &
  sleep 15
  if up $log && readback $log; then
    CACHE=4:0.16:3 python $V/sglang_case.py case=$V/case_ir.txt task=ref2va \
      tag=ab$name 768:25:121 2>&1 | tee -a $R
    stages $log
  fi
  stop; }

stop
want A0 && arm A0 --layerwise-offload-components text_encoder --image-encoder-cpu-offload --vae-cpu-offload
want A1 && arm A1 --layerwise-offload-components text_encoder --image-encoder-cpu-offload
want A2 && arm A2 --layerwise-offload-components text_encoder
want A3 && arm A3
echo OFFLOAD_DONE | tee -a $R
