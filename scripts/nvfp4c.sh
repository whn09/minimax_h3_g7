#!/usr/bin/env bash
# NVFP4, in the layout this image's loader actually asserts -- and the low-TP arms it unlocks.
#
#   kubectl exec h3-serve -- bash -lc 'cd /data/h3 && ARMS="C N4" setsid nohup bash nvfp4c.sh \
#     > /data/h3/sglang/logs/nvfp4c_run.log 2>&1 &'
#
# WHY. The two parked NVFP4 arms rendered noise (G7.md 3.2.1) and the parked hypothesis was tensor
# parallelism. That hypothesis is wrong. `nvfp4_comfy_layout.py`'s CHECK mode dequantized the file the
# way this image's loader will and got **cos = -0.000, rel = 1.45** against the original bf16 weights,
# on both a qkv and a non-qkv layer -- i.e. the weights the DiT computed with were unrelated to the
# model, at any TP. The loader hard-codes swizzled scales, swapped nibbles and native qkv rows for any
# nvfp4-marked H3 checkpoint; our file has none of the three. Same CHECK after the transform: **cos =
# 1.000, rel = 0.095**, the group-16 e2m1 floor. See that script's header for the file:line trail.
#
# C IS THE GATE AND N4 IS THE CONTROL. C refuses to write unless it reordered exactly 52 qkv tensors,
# quantized-layer count is 208, and the checked layers come back within 0.12 of the bf16 reference. N4
# then serves the converted file at the SAME topology as the noise arm (TP=4 x U=2, sage, 768p, 121 f,
# seed 42, 25 steps, ref short edge 2048), where we have both a number (125.33 s) and a watched render
# to compare against. A clean render at ~125 s says the diagnosis and the fix are both right, and that
# NVFP4's 7 % over fp8 is real rather than the artefact of computing on garbage.
#
# N5 IS THE ASK: TP=1 x ULYSSES=8. It is only reachable at all with --dit-layerwise-offload, because
# 37.5 GB of weights unsharded does not fit on a 32 GB card -- TP is the only thing that shards
# weights, Ulysses shards activations. Offline fp8 already showed streaming hides behind compute at
# TP=2 (F3, G7.md 3.1.2), and this doubles the streamed bytes per card per step, so this arm measures
# whether that stays true when the last all-reduce is gone. N6 is TP=2, the topology that won for fp8,
# which is the one likely to be fastest; N5 and N6 together say whether the collective saving or the
# streaming cost dominates.
set -uo pipefail
V=/data/h3; L=$V/sglang/logs; R=$L/nvfp4c.txt
export ROOT=$V/sglang VDNROOT=$V FRAMES=121 OUTDIR=$V/pull/case REFDIR=$V/ref
S=$V/hf/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
F=$V/nvfp4c_ref2va.safetensors
FB=$V/nvfp4c_fl2va.safetensors
mkdir -p "$L"
echo "########## $(date +%F' '%T)  ARMS=${ARMS:-<default>}" >> $R
SAGE=(--attention-backend sage_attn
      --component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa)
# Same two reasons as fp8off.sh: the RUNAI streamer drops the safetensors metadata header, which is the
# only thing marking these tensors nvfp4, and QUANT= (empty) keeps --quantization off the command line
# so the file's own header is the single source of the precision.
export SGLANG_USE_RUNAI_MODEL_STREAMER=0
unset SGLANG_CACHE_DIT_ENABLED

up() { local log=$1 i; for i in $(seq 1 40); do
  grep -q "fired up and ready to roll" "$log" 2>/dev/null && { echo "  ready ${i}0s" | tee -a $R; return 0; }
  grep -q "Error while loading component\|Server warmup failed\|processing failed" "$log" 2>/dev/null && { echo "  FAILED" | tee -a $R; grep -m3 -i "outofmemory\|Error\|ValueError\|RuntimeError" "$log" | cut -c1-260 | tee -a $R; return 1; }
  pgrep -f '[s]glang.*serve' >/dev/null || { echo "  SERVER GONE" | tee -a $R; tail -6 "$log" | cut -c1-260 | tee -a $R; return 1; }
  sleep 10; done; echo "  TIMEOUT" | tee -a $R; return 1; }
stop() { pkill -f '[s]glang.*serve' >/dev/null 2>&1; sleep 12; }
want() { case " ${ARMS:-C N4} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Sage must be active or the number means nothing (torch_sdpa is the 177 s baseline), and the quant
# config name must show up or the file loaded as raw bytes. NOT a quality gate: the render is the only
# thing that can tell noise from signal, which is why every arm here has to be watched.
readback() { local log=$1 line
  line=$(grep -h "Using .* attention backend" "$log" | tail -1)
  echo "  attention: ${line:-<none found>}" | tee -a $R
  grep -hio "comfy_nvfp4\|modelopt_fp4\|nvfp4" "$log" | sort | uniq -c | sed 's/^/  quant: /' | tee -a $R
  case "$line" in
    *sage_attn*) return 0 ;;
    *) echo "  !! SAGE NOT ACTIVE -- skipping the render." | tee -a $R; return 1 ;;
  esac; }

# One convert arm per partition. Pure CPU, ~3 min, peak host RAM about one file (save_file wants every
# tensor at once) against 671 GB free. BF16= is not optional: the script refuses to write a file it
# could not verify against the original weights, which is the whole lesson of the parked arms.
convert() { local v=$1 out=$2
  echo "=== convert $v -> $out  $(date +%T)" | tee -a $R
  SRC=$V/nvfp4_$v.safetensors DST=$out BF16=$S/${3}/transformer HEADS=56 HEAD_DIM=128 \
    python3 $V/nvfp4_comfy_layout.py 2>&1 | grep -v "Flax classes" | tee -a $R
  local rc=${PIPESTATUS[0]}
  echo "  exit $rc  $(date +%T)" | tee -a $R
  ls -l "$out" 2>/dev/null | tee -a $R
  [ "$rc" = 0 ] || { echo "CONVERT FAILED -- not serving a file that failed its own check" | tee -a $R
                     echo NVFP4C_DONE | tee -a $R; exit 1; }; }

stop
want C  && convert ref2va "$F"  Ref2VA
want CB && convert fl2va  "$FB" FL2VA

if want N4; then
echo "=== N4  CONTROL: converted NVFP4, TP=4 x U=2   (vs the noise arm's 125.33 s, and fp8's 134.27 s)" | tee -a $R
log=$L/serve_ref2va_768p_nvfp4ctp4.log; rm -f $log
QUANT= GPUS=8 TP=4 ULYSSES=2 LOGTAG=nvfp4ctp4 \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
    --transformer-weights-path $F "${SAGE[@]}" \
    > $L/launch_nvfp4c_a.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=nvfp4ctp4 768:25:121 2>&1 | tee -a $R
fi
stop
fi

if want N5; then
echo "=== N5  THE ASK: converted NVFP4, TP=1 x U=8 + --dit-layerwise-offload" | tee -a $R
# TP=1 removes every all-reduce from the linears; Ulysses' all-to-all in attention remains. The bill
# for it is that all 37.5 GB streams per card per step instead of 18.7 GB at TP=2.
log=$L/serve_ref2va_768p_nvfp4ctp1.log; rm -f $log
QUANT= GPUS=8 TP=1 ULYSSES=8 LOGTAG=nvfp4ctp1 \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
    --transformer-weights-path $F \
    --dit-layerwise-offload --layerwise-offload-components text_encoder \
    --image-encoder-cpu-offload --vae-cpu-offload "${SAGE[@]}" \
    > $L/launch_nvfp4c_b.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=nvfp4ctp1 768:25:121 2>&1 | tee -a $R
fi
stop
fi

if want N6; then
echo "=== N6  converted NVFP4, TP=2 x U=4 + --dit-layerwise-offload   (fp8's winning topology: 119.43 s)" | tee -a $R
log=$L/serve_ref2va_768p_nvfp4ctp2.log; rm -f $log
QUANT= GPUS=8 TP=2 ULYSSES=4 LOGTAG=nvfp4ctp2 \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
    --transformer-weights-path $F \
    --dit-layerwise-offload --layerwise-offload-components text_encoder \
    --image-encoder-cpu-offload --vae-cpu-offload "${SAGE[@]}" \
    > $L/launch_nvfp4c_c.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=nvfp4ctp2 768:25:121 2>&1 | tee -a $R
fi
stop
fi

# t2va, at whichever of N5/N6 won. TP is passed in rather than hard-coded so this arm does not have to
# be edited when the answer changes: NVTP=1 NVUP=8 ARMS=N7. Same prompt as G1 (case_t2va_v2.txt @wide).
if want N7; then
echo "=== N7  t2va@wide, converted NVFP4, TP=${NVTP:-2} x U=${NVUP:-4}   (vs offline fp8's 74.65 s)" | tee -a $R
log=$L/serve_base_768p_nvfp4c.log; rm -f $log
QUANT= GPUS=8 TP=${NVTP:-2} ULYSSES=${NVUP:-4} LOGTAG=nvfp4c \
  setsid nohup bash $V/sglang_base_arm.sh serve 768 \
    --transformer-weights-path $FB \
    --dit-layerwise-offload --layerwise-offload-components text_encoder \
    --image-encoder-cpu-offload --vae-cpu-offload "${SAGE[@]}" \
    > $L/launch_nvfp4c_d.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  python $V/sglang_case.py case=$V/case_t2va_v2.txt task=t2va label=wide \
    tag=nvfp4c 768:25:121 2>&1 | tee -a $R
fi
stop
fi

if want N8; then
echo "=== N8  N5 + Cache-DiT   (the composed floor: NVFP4 + TP=1 + sage + caching)" | tee -a $R
# Same shape as fp8off.sh's F5 and for the same reason: enable_cache_dit is per-request, so the `off`
# request in the SAME process is the control that proves the process still reaches 101 s. Without it a
# fast cached number could just be a warmer machine. rdt 0.16 only -- 0.10 was strictly slower on both
# tasks (G7.md 3.1.2) and this is a third approximation stacked on the same render, so the arm exists to
# be watched, not to be swept.
export SGLANG_CACHE_DIT_ENABLED=true
log=$L/serve_ref2va_768p_nvfp4ctp1cd.log; rm -f $log
QUANT= GPUS=8 TP=1 ULYSSES=8 LOGTAG=nvfp4ctp1cd \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
    --transformer-weights-path $F \
    --dit-layerwise-offload --layerwise-offload-components text_encoder \
    --image-encoder-cpu-offload --vae-cpu-offload "${SAGE[@]}" \
    > $L/launch_nvfp4c_e.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  for a in "off cdoff" "4:0.16:3 cd016"; do
    set -- $a
    CACHE=$1 python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=nvfp4ctp1$2 768:25:121 2>&1 | tee -a $R
    grep -h "Collected Context Config\|cache-dit enabled on transformer" $log | tail -2 \
      | cut -c1-200 | sed 's/^/  cache: /' | tee -a $R
  done
fi
stop
unset SGLANG_CACHE_DIT_ENABLED
fi

if want N9; then
echo "=== N9  N7 + Cache-DiT   (t2va half of the composed floor; vs offline fp8's 38.66 s)" | tee -a $R
# The last cell of the table. N7 already measured this topology uncached in its own process, but the
# `off` request is repeated here anyway: a 2x claim has to be against a control from the SAME process,
# which is the rule that caught the fictitious 61.21 s T3 (G7.md 4.1).
export SGLANG_CACHE_DIT_ENABLED=true
log=$L/serve_base_768p_nvfp4ccd.log; rm -f $log
QUANT= GPUS=8 TP=1 ULYSSES=8 LOGTAG=nvfp4ccd \
  setsid nohup bash $V/sglang_base_arm.sh serve 768 \
    --transformer-weights-path $FB \
    --dit-layerwise-offload --layerwise-offload-components text_encoder \
    --image-encoder-cpu-offload --vae-cpu-offload "${SAGE[@]}" \
    > $L/launch_nvfp4c_f.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  for a in "off cdoff" "4:0.16:3 cd016"; do
    set -- $a
    CACHE=$1 python $V/sglang_case.py case=$V/case_t2va_v2.txt task=t2va label=wide \
      tag=nvfp4ccd$2 768:25:121 2>&1 | tee -a $R
    grep -h "Collected Context Config\|cache-dit enabled on transformer" $log | tail -2 \
      | cut -c1-200 | sed 's/^/  cache: /' | tee -a $R
  done
fi
stop
unset SGLANG_CACHE_DIT_ENABLED
fi
echo NVFP4C_DONE | tee -a $R
