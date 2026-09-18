#!/usr/bin/env bash
# Cache-DiT on g7, and a TP=2 retry, at fp8 + SageAttention.
#
#   kubectl exec h3-serve -- bash -lc 'setsid nohup bash /data/h3/cachedit.sh > /data/h3/logs/cd.txt 2>&1 &'
#   ARMS="C0 C1 C2" bash /data/h3/cachedit.sh      # ref2va only
#
# WHY THIS IS RUNNABLE AT ALL, GIVEN §4 SAID IT WAS NOT. Three different switches ask for Cache-DiT
# and only one of them is the audited, hardware-gated one:
#   quality="high"            -> the vendor's audited (4, 0.04, 1) preset. REFUSED on g7: the audit
#                                pins 4xH200 / fl2va / unquantized / TP=1 x U=4 plus the device string.
#   --cache-dit-config        -> "Enables cache-dit for diffusers backend" (its own help text). H3 runs
#                                the NATIVE pipeline, so this flag is what produced the byte-identical
#                                105.40 s no-op. It was never a Cache-DiT measurement.
#   SGLANG_CACHE_DIT_ENABLED  -> generic mode, knob-driven, NOT gated. constants.py, verbatim:
#   + per-request              "Process-wide SGLANG_CACHE_DIT_* environment controls remain available
#                               for manual experiments and are independent of this field."
# So the env turns the machinery on and each request carries its own knobs. That is the whole reason
# ONE server can serve the baseline and both cached arms: `enable_cache_dit` is per request, so the
# three numbers below are matched to the same process, the same weights and the same seed.
#
# THE GATE THAT BITES IF YOU SEND BOTH. minimax_h3/stages/denoising.py:
#   generic_requested = generic_enabled and "quality" not in explicit_fields
# A request carrying `quality` at all cannot get generic mode -- it gets the audited path, which is
# refused here. sglang_case.py therefore sends `quality` OR the cache knobs, never both.
#
# ONE UNVALIDATED COMBINATION, STATED SO IT IS NOT DISCOVERED LATER. server_args.py:3766 warns
# "cache-dit is enabled with hybrid parallelism (SP + TP). Proceeding anyway (SGLang integration may
# support this mode)" whenever SP>1 and TP>1 -- which is exactly TP=4 x U=2. A warning, not a refusal.
# It is the reason C0 is re-rendered here instead of quoting the 134.29 s already in G7.md 2.1: if the
# hybrid path is subtly wrong, the baseline from the SAME process is the only thing that shows it.
#
# THE THREE ARMS AND WHY THESE THRESHOLDS. residual_diff_threshold is the entire tradeoff and the
# useful range spans 6x, so one value would prove nothing:
#   C0  cache=off      the in-session baseline. Also the file the user compares against by eye.
#   C1  cache=4:0.04:1 the vendor's own audited numbers, transcribed exactly (constants.py:64
#                      MINIMAX_H3_HIGH_QUALITY_CACHE_DIT_CONFIG = (4, 0.04, 1), "Measured SSIM 0.931 /
#                      PSNR 28.16 dB against quality=lossless"). Conservative; expect a small gain.
#                      Its value is that the quality point is KNOWN on other hardware, so a bad render
#                      here is evidence about g7, not about the knobs.
#   C2  cache=4:0.16:3 the g7e project's measured 768p threshold, at the env default of 3 consecutive
#                      cached steps. This is where their 1.6x-2.1x lives. Expect the visible loss here.
# Fn=1 / Bn=0 / taylorseer=off are left at the env defaults, which are already the audited preset's.
#
# HELD FIXED against every arm in G7.md 2.1: base model, 768p, 121 f (5.04 s), seed 42, 25 steps,
# fp8 + sage, TP=4 x ULYSSES=2, reference short edge 2048. t2va uses case_t2va_v2.txt @wide.
set -uo pipefail
V=/data/h3; L=$V/sglang/logs; R=$L/cachedit.txt
export ROOT=$V/sglang VDNROOT=$V FRAMES=121 OUTDIR=$V/pull/case REFDIR=$V/ref
: > $R
SAGE=(--attention-backend sage_attn
      --component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa)
# This is the switch. Without it every request below is silently lossless and all three arms return
# the same number -- the exact failure the --cache-dit-config arm already walked into once.
export SGLANG_CACHE_DIT_ENABLED=true

up() { local log=$1 i; for i in $(seq 1 40); do
  grep -q "fired up and ready to roll" "$log" 2>/dev/null && { echo "  ready ${i}0s" | tee -a $R; return 0; }
  grep -q "Error while loading component\|Server warmup failed\|processing failed" "$log" 2>/dev/null && { echo "  FAILED" | tee -a $R; grep -m3 -i "outofmemory\|Error" "$log" | cut -c1-260 | tee -a $R; return 1; }
  pgrep -f '[s]glang.*serve' >/dev/null || { echo "  SERVER GONE" | tee -a $R; tail -6 "$log" | cut -c1-260 | tee -a $R; return 1; }
  sleep 10; done; echo "  TIMEOUT" | tee -a $R; return 1; }
stop() { pkill -f '[s]glang.*serve' >/dev/null 2>&1; sleep 12; }
want() { case " ${ARMS:-C0 C1 C2 D0 D1 D2 T1 T2} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Both readbacks come out of the server's own log because both failure modes are silent. Sage falling
# back to torch_sdpa reproduces the 105/177 s baseline; Cache-DiT not mounting reproduces the
# lossless number, which is exactly what a "no gain" result looks like. The cache line is reported,
# not asserted, because its absence before the first request is normal -- the hook mounts per batch.
readback() { local log=$1 line
  line=$(grep -h "Using .* attention backend" "$log" | tail -1)
  echo "  attention: ${line:-<none found>}" | tee -a $R
  case "$line" in
    *sage_attn*) return 0 ;;
    *) echo "  !! SAGE NOT ACTIVE -- skipping the renders." | tee -a $R; return 1 ;;
  esac; }
cachecheck() { grep -hi "cache.dit\|cache_dit\|DBCache\|hybrid parallelism" "$1" | tail -6 \
  | cut -c1-220 | sed 's/^/  cache: /' | tee -a $R; }

stop

# ---------------------------------------------------------------- ref2va (the arm that matters)
if want C0 || want C1 || want C2; then
echo "=== ref2va  fp8 + sage,  SGLANG_CACHE_DIT_ENABLED=true   (lossless reference: 134.29 s)" | tee -a $R
log=$L/serve_ref2va_768p_cd.log; rm -f $log
GPUS=8 TP=4 ULYSSES=2 LOGTAG=cd \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 "${SAGE[@]}" \
    > $L/launch_cd_ref.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  for a in "C0 off cdoff" "C1 4:0.04:1 cd004" "C2 4:0.16:3 cd016"; do
    set -- $a; want $1 || continue
    echo "--- $1  cache=$2" | tee -a $R
    CACHE=$2 python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=$3 768:25:121 2>&1 | tee -a $R
    cachecheck $log
  done
fi
stop
fi

# ---------------------------------------------------------------- t2va@wide, same three
if want D0 || want D1 || want D2; then
echo "=== t2va@wide  fp8 + sage,  SGLANG_CACHE_DIT_ENABLED=true   (lossless reference: 85.64 s)" | tee -a $R
log=$L/serve_base_768p_cd.log; rm -f $log
GPUS=8 TP=4 ULYSSES=2 LOGTAG=cd \
  setsid nohup bash $V/sglang_base_arm.sh serve 768 "${SAGE[@]}" \
    > $L/launch_cd_base.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  for a in "D0 off cdoff" "D1 4:0.04:1 cd004" "D2 4:0.16:3 cd016"; do
    set -- $a; want $1 || continue
    echo "--- $1  cache=$2" | tee -a $R
    CACHE=$2 python $V/sglang_case.py case=$V/case_t2va_v2.txt task=t2va label=wide tag=$3 768:25:121 2>&1 | tee -a $R
    cachecheck $log
  done
fi
stop
fi

# ---------------------------------------------------------------- TP=2, retried
# WHY RETRY SOMETHING G7.md 3.1 ALREADY CALLS DEAD. Two reasons and they are both narrow. (1) The
# image has moved a month (20518d85, 2026-09-18) since those arms, and the failure was a warmup
# allocation, which is the kind of thing that moves. (2) The arithmetic says the blocker is WEIGHTS,
# not activations, and T2 is the one configuration that attacks weights and was never run.
#
# THE ARITHMETIC, so the result is interpretable either way. Activation residency is ~11 GiB/card at
# BOTH topologies, because TP and Ulysses shard different axes and multiply to the same 1/8 -- so
# halving TP buys nothing on activations and costs 15.9 GB/card of weights. T1 = 18.7 (adaln-online
# bf16 weights) + 2.3 (the "64 plans x 4 timesteps" rebuild slab) + ~11 = ~32 against 31.37 GiB
# usable, i.e. it misses by about a gigabyte, which is why it OOMed at 30.52 GiB last time.
# T2 adds --dit-layerwise-offload, which streams the DiT's weights per layer instead of holding them,
# and is therefore the only lever that touches the term that is actually too big.
#
# T2 CANNOT WIN ON SPEED AND IS NOT MEANT TO. Streaming ~37 GB of DiT weights from host RAM once per
# step over PCIe adds seconds per step; bf16 already costs +19.7 %/step against fp8, against a
# 10-15 % topology gain from halving TP. The question being answered is the user's literal one --
# does TP=2 run at all -- not whether it is faster. If T2 renders, the number goes in the table
# marked as a feasibility point, not as a candidate.
if want T1; then
echo "=== T1  ref2va bf16 TP=2 x U=4, adaln-online + all offloads   (retry of G7.md 3.1)" | tee -a $R
log=$L/serve_ref2va_768p_tp2.log; rm -f $log
QUANT= GPUS=8 TP=2 ULYSSES=4 LOGTAG=tp2 \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
    --minimax-h3-adaln-online --layerwise-offload-components text_encoder \
    --image-encoder-cpu-offload --vae-cpu-offload "${SAGE[@]}" \
    > $L/launch_tp2_a.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=tp2 768:25:121 2>&1 | tee -a $R
fi
stop
fi

if want T2; then
echo "=== T2  same + --dit-layerwise-offload   (the only lever on the term that is too big)" | tee -a $R
log=$L/serve_ref2va_768p_tp2off.log; rm -f $log
QUANT= GPUS=8 TP=2 ULYSSES=4 LOGTAG=tp2off \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 \
    --minimax-h3-adaln-online --dit-layerwise-offload \
    --layerwise-offload-components text_encoder \
    --image-encoder-cpu-offload --vae-cpu-offload "${SAGE[@]}" \
    > $L/launch_tp2_b.log 2>&1 < /dev/null &
sleep 15
if up $log && readback $log; then
  python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=tp2off 768:25:121 2>&1 | tee -a $R
fi
stop
fi
echo CACHEDIT_DONE | tee -a $R
