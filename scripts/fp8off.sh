#!/usr/bin/env bash
# Offline-FP8 checkpoint arms: the only route to fp8 below TP=4 on a 32 GB card.
#
#   kubectl exec h3-serve -- bash -lc 'cd /data/h3 && ARMS="Q F1 F2" setsid nohup bash fp8off.sh \
#     > /data/h3/sglang/logs/fp8off_run.log 2>&1 &'
#
# WHY. Online fp8 cannot load at TP=2 -- the loader lands the 65.65 GiB bf16 checkpoint on the cards
# and casts there (32.8 GiB/card), and arm T4 showed --dit-layerwise-offload does not rescue it
# because the quantized load path materialises parameters on device to attach weight scales. A
# pre-quantized file needs no cast, so it lands 22.5 GiB/card at TP=2 with nothing streaming. See
# G7.md 3.1.1 and fp8_quantize_transformer.py's header for the format and the qkv-reorder trap.
#
# F1 IS A CONTROL AND IT IS NOT OPTIONAL. The failure mode of a wrong offline quantization is a clean
# render of garbage, with no error (this is exactly how the NVFP4 arms burned an afternoon). So the
# file is first served at the SAME topology as the online-fp8 arm we already have ground truth for:
# TP=4 x U=2, sage, 768p, 121 f, seed 42, 25 steps, ref short edge 2048 -> 134.27 s and a watched
# render. If F1 comes back near 134 s and its video matches, the file is right and F2 is a pure
# topology change. If F1 is garbage, F2 would be garbage too and would prove nothing about TP=2.
#
# F2 IS THE QUESTION. Same file, TP=2 x U=4, no DiT streaming. 22.5 GiB/card of weights plus ~11 GiB
# of activations is 33.5 against 31.37, so the two cheap offloads (image encoder, VAE) are on to buy
# the difference back. If it still OOMs, F3 adds --dit-layerwise-offload, which is guaranteed to load
# (T3 did it at bf16) at the cost of per-step weight streaming.
#
# ref2va ONLY. The user's standing instruction is "你主要看ref2va就行，预算有限" -- and ref2va is also
# the harder arm (it holds the reference tower at short edge 2048), so it is the one that decides
# whether the topology fits.
set -uo pipefail
V=/data/h3; L=$V/sglang/logs; R=$L/fp8off.txt
export ROOT=$V/sglang VDNROOT=$V FRAMES=121 OUTDIR=$V/pull/case REFDIR=$V/ref
S=$V/hf/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
F=$V/fp8_ref2va.safetensors
mkdir -p "$L"
# APPEND, do not truncate. These arms get run in batches (ARMS="Q F1 F2", then F4/F3, then F5) as each
# result decides the next one, and a `: >` here silently threw away the F1-F4 table on the third batch.
echo "########## $(date +%F' '%T)  ARMS=${ARMS:-<default>}" >> $R
SAGE=(--attention-backend sage_attn
      --component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa)
# RUNAI streamer off for the same reason as the NVFP4 arms: it streams shards straight to device and
# does not carry the safetensors metadata header through, and that header is the ONLY thing telling
# the loader these tensors are fp8 rather than raw bytes.
export SGLANG_USE_RUNAI_MODEL_STREAMER=0
# Cache-DiT explicitly OFF for the whole process. These arms are precision/topology measurements and
# have to be comparable to the lossless numbers; leaving SGLANG_CACHE_DIT_ENABLED unset is what does
# that (its default is False, envs.py:64), and no request here sends cache fields.
unset SGLANG_CACHE_DIT_ENABLED

up() { local log=$1 i; for i in $(seq 1 40); do
  grep -q "fired up and ready to roll" "$log" 2>/dev/null && { echo "  ready ${i}0s" | tee -a $R; return 0; }
  grep -q "Error while loading component\|Server warmup failed\|processing failed" "$log" 2>/dev/null && { echo "  FAILED" | tee -a $R; grep -m3 -i "outofmemory\|Error\|ValueError\|RuntimeError" "$log" | cut -c1-260 | tee -a $R; return 1; }
  pgrep -f '[s]glang.*serve' >/dev/null || { echo "  SERVER GONE" | tee -a $R; tail -6 "$log" | cut -c1-260 | tee -a $R; return 1; }
  sleep 10; done; echo "  TIMEOUT" | tee -a $R; return 1; }
stop() { pkill -f '[s]glang.*serve' >/dev/null 2>&1; sleep 12; }
want() { case " ${ARMS:-Q F1 F2} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Two readbacks, because both failures are silent. No sage falls back to torch_sdpa (= the 177 s
# baseline, not the 134 s one), and a checkpoint whose metadata header was dropped serves as bf16 --
# which at TP=2 would OOM but at TP=4 would just be a slow "success". comfy_fp8 is the config name
# ComfyFp8Config.get_name() returns, so seeing it is proof the marker set parsed as ["float8_e4m3fn"].
readback() { local log=$1 line
  line=$(grep -h "Using .* attention backend" "$log" | tail -1)
  echo "  attention: ${line:-<none found>}" | tee -a $R
  grep -hio "comfy_fp8\|float8_e4m3fn\|is_checkpoint_fp8_serialized[^,]*" "$log" | sort | uniq -c \
    | sed 's/^/  quant: /' | tee -a $R
  case "$line" in
    *sage_attn*) return 0 ;;
    *) echo "  !! SAGE NOT ACTIVE -- skipping the render." | tee -a $R; return 1 ;;
  esac; }

stop
if want Q; then
echo "=== Q  export fp8_ref2va.safetensors  $(date +%T)" | tee -a $R
# Pure CPU, ~2 min on this box, peak host RAM about one output file against 726 GB. Deliberately not
# run alongside a timed request: a 32-thread torch job on the same host skews any inference number next
# to it. Exits non-zero unless it quantized exactly 208 layers AND reordered exactly 52 qkv tensors.
SRC=$S/Ref2VA/transformer DST=$F HEADS=56 HEAD_DIM=128 \
  python3 $V/fp8_quantize_transformer.py 2>&1 | tee -a $R
rc=${PIPESTATUS[0]}
echo "  exit $rc  $(date +%T)" | tee -a $R
ls -l "$F" 2>/dev/null | tee -a $R
[ "$rc" = 0 ] || { echo "EXPORT FAILED -- not serving a file that failed its own check" | tee -a $R
                   echo FP8OFF_DONE | tee -a $R; exit 1; }
fi

if want F1; then
echo "=== F1  CONTROL: offline fp8, TP=4 x U=2   (vs online fp8 134.27 s, same everything else)" | tee -a $R
log=$L/serve_ref2va_768p_offp8tp4.log; rm -f $log
# QUANT= (empty) keeps --quantization off the command line entirely: the precision comes from the
# file's own metadata header, and passing the flag would build a second config that wins.
QUANT= GPUS=8 TP=4 ULYSSES=2 LOGTAG=offp8tp4 \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
    --transformer-weights-path $F "${SAGE[@]}" \
    > $L/launch_offp8_a.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=offp8tp4 768:25:121 2>&1 | tee -a $R
fi
stop
fi

if want F2; then
echo "=== F2  THE QUESTION: offline fp8, TP=2 x U=4, no DiT streaming" | tee -a $R
log=$L/serve_ref2va_768p_offp8tp2.log; rm -f $log
QUANT= GPUS=8 TP=2 ULYSSES=4 LOGTAG=offp8tp2 \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
    --transformer-weights-path $F \
    --image-encoder-cpu-offload --vae-cpu-offload "${SAGE[@]}" \
    > $L/launch_offp8_b.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=offp8tp2 768:25:121 2>&1 | tee -a $R
fi
stop
fi

if want F4; then
echo "=== F4  F2 + text_encoder layerwise offload   (F2 loaded fine and OOMed 64 MB into the forward)" | tee -a $R
# F2's failure is worth reading precisely: "Error executing request None ... 30.47 GiB allocated,
# tried to allocate 64.00 MiB" is the WARMUP FORWARD, not the load. So 22.5 GiB/card of offline-fp8
# weights fits at TP=2 -- which is the thing online fp8 could never do -- and what is missing is about
# a gigabyte of ACTIVATION headroom. That makes the right lever the cheapest one available:
# --layerwise-offload-components text_encoder frees the resident text tower and costs latency ONCE PER
# REQUEST rather than once per step, unlike --dit-layerwise-offload (F3). nvfp4.sh's N3 arm already
# uses it without --dit-layerwise-offload, so the two flags are independent.
log=$L/serve_ref2va_768p_offp8tp2te.log; rm -f $log
QUANT= GPUS=8 TP=2 ULYSSES=4 LOGTAG=offp8tp2te \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
    --transformer-weights-path $F \
    --layerwise-offload-components text_encoder \
    --image-encoder-cpu-offload --vae-cpu-offload "${SAGE[@]}" \
    > $L/launch_offp8_d.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=offp8tp2te 768:25:121 2>&1 | tee -a $R
fi
stop
fi

if want F3; then
echo "=== F3  fallback: F2 + --dit-layerwise-offload   (only worth running if F2 OOMed)" | tee -a $R
log=$L/serve_ref2va_768p_offp8tp2lw.log; rm -f $log
QUANT= GPUS=8 TP=2 ULYSSES=4 LOGTAG=offp8tp2lw \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
    --transformer-weights-path $F \
    --dit-layerwise-offload --layerwise-offload-components text_encoder \
    --image-encoder-cpu-offload --vae-cpu-offload "${SAGE[@]}" \
    > $L/launch_offp8_c.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=offp8tp2lw 768:25:121 2>&1 | tee -a $R
fi
stop
fi

if want F5; then
echo "=== F5  F3 + Cache-DiT   (do the topology win and the caching win compose?)" | tee -a $R
# F3 is 119.43 s lossless and 4.78 s/step -- the best exact-math number on this box. Cache-DiT at
# rdt=0.16 was worth 2.16x on top of the OLD best (134.27 -> 62.30 s, G7.md 4.1). Whether the two
# compose is not obvious: Cache-DiT skips transformer blocks, and F3 STREAMS those blocks from host
# RAM, so a skipped block may or may not also skip its copy. All three requests run in ONE server
# process because enable_cache_dit is per-request, and the `off` request is the control that proves
# the process still reaches 119 s -- without it a fast cached number could just be a faster machine.
export SGLANG_CACHE_DIT_ENABLED=true
log=$L/serve_ref2va_768p_offp8tp2cd.log; rm -f $log
QUANT= GPUS=8 TP=2 ULYSSES=4 LOGTAG=offp8tp2cd \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
    --transformer-weights-path $F \
    --dit-layerwise-offload --layerwise-offload-components text_encoder \
    --image-encoder-cpu-offload --vae-cpu-offload "${SAGE[@]}" \
    > $L/launch_offp8_e.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  for a in "off cdoff" "4:0.10:2 cd010" "4:0.16:3 cd016"; do
    set -- $a
    CACHE=$1 python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=offp8tp2$2 768:25:121 2>&1 | tee -a $R
    # Same readback as cachedit.sh: the mode that actually fired, straight from the server's log, so
    # an inherited env default (rdt=0.24) cannot be reported as a knob we chose.
    grep -h "Collected Context Config\|cache-dit enabled on transformer" $log | tail -2 \
      | cut -c1-200 | sed 's/^/  cache: /' | tee -a $R
  done
fi
stop
unset SGLANG_CACHE_DIT_ENABLED
fi

# ---------------------------------------------------------------------------------------------------
# t2va. Everything above is ref2va, which is where the standing instruction points ("你主要看ref2va就行")
# and which is the harder arm on memory. But the fp8/TP=2 result is a property of the DiT, not of the
# task, so it has to be shown on the other partition too before it goes in the document as a floor.
# t2va uses case_t2va_v2.txt @wide, not case_ir.txt -- see sage.sh's header for why that prompt.
# ---------------------------------------------------------------------------------------------------
FB=$V/fp8_fl2va.safetensors
if want QB; then
echo "=== QB  export fp8_fl2va.safetensors  $(date +%T)" | tee -a $R
# fp8_fl2va, not "fp8_t2va": MINIMAX_H3_TASK_PARTITIONS maps t2va -> fl2va, so the base server holds
# the FL2VA weight partition and that is the transformer being replaced.
SRC=$S/FL2VA/transformer DST=$FB HEADS=56 HEAD_DIM=128 \
  python3 $V/fp8_quantize_transformer.py 2>&1 | tee -a $R
rc=${PIPESTATUS[0]}
echo "  exit $rc  $(date +%T)" | tee -a $R
ls -l "$FB" 2>/dev/null | tee -a $R
[ "$rc" = 0 ] || { echo "EXPORT FAILED -- not serving a file that failed its own check" | tee -a $R
                   echo FP8OFF_DONE | tee -a $R; exit 1; }
fi

if want G1; then
echo "=== G1  t2va@wide, offline fp8, TP=2 x U=4 (+Cache-DiT)  (vs online fp8 TP=4: 86.03 s)" | tee -a $R
# QUANT= IS NOT OPTIONAL AND IT IS NOT THE SAME AS THE ref2va ARMS. sglang_base_arm.sh:52 defaults
# QUANT to *fp8*, the opposite of sglang_ref2va_arm.sh:58, so leaving it unset here would turn ONLINE
# quantization back on underneath an already-quantized file. It also defaults TP=1, hence both knobs.
export SGLANG_CACHE_DIT_ENABLED=true
log=$L/serve_base_768p_offp8tp2cd.log; rm -f $log
QUANT= GPUS=8 TP=2 ULYSSES=4 LOGTAG=offp8tp2cd \
  setsid nohup bash $V/sglang_base_arm.sh serve 768 \
    --transformer-weights-path $FB \
    --dit-layerwise-offload --layerwise-offload-components text_encoder \
    --image-encoder-cpu-offload --vae-cpu-offload "${SAGE[@]}" \
    > $L/launch_offp8_f.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  for a in "off cdoff" "4:0.16:3 cd016"; do
    set -- $a
    CACHE=$1 python $V/sglang_case.py case=$V/case_t2va_v2.txt task=t2va label=wide \
      tag=offp8tp2$2 768:25:121 2>&1 | tee -a $R
    grep -h "Collected Context Config\|cache-dit enabled on transformer" $log | tail -2 \
      | cut -c1-200 | sed 's/^/  cache: /' | tee -a $R
  done
fi
stop
unset SGLANG_CACHE_DIT_ENABLED
fi

if want F6; then
echo "=== F6  F3 WITHOUT sage   (the quality gate for sage, at the topology we actually ship)" | tee -a $R
# "sage会带来精度损失吗?" -- yes, it quantizes the attention computation, so it is an approximation and
# not a free win. Every other arm in this file has it on, which means none of them can answer the
# question: they are all on the same side of it. F6 is F3 with the attention backend as the ONLY
# change, so ref2va_offp8_tp2.mp4 and this render are a matched pair for watching, and the timing
# difference is sage's speedup measured at TP=2 rather than inherited from the TP=4 sweep (1.32x).
# No --attention-backend at all: the arm scripts' default is torch_sdpa, which is the unapproximated
# attention path.
# NOTE THIS IS THE ONE ARM WHERE OMITTING "${SAGE[@]}" IS DELIBERATE. Everywhere else its absence is
# a bug -- readback() exists precisely to catch a silent fallback to sdpa, which reads as a 32% slower
# "success". Here the readback would fail the arm for doing the right thing, so it is skipped and the
# backend line is printed for the record instead.
log=$L/serve_ref2va_768p_offp8tp2nosage.log; rm -f $log
QUANT= GPUS=8 TP=2 ULYSSES=4 LOGTAG=offp8tp2nosage \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
    --transformer-weights-path $F \
    --dit-layerwise-offload --layerwise-offload-components text_encoder \
    --image-encoder-cpu-offload --vae-cpu-offload \
    > $L/launch_offp8_g.log 2>&1 < /dev/null &
sleep 15
if up $log; then
  grep -h "Using .* attention backend" $log | tail -1 | sed 's/^/  attention: /' | tee -a $R
  python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=offp8tp2nosage 768:25:121 2>&1 | tee -a $R
fi
stop
fi
echo FP8OFF_DONE | tee -a $R
