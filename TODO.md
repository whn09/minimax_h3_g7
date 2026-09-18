# Parked work on g7.48xlarge

## NVFP4 — parked 2026-09-18. It loads, it is 7–9 % faster than sage alone, and it renders noise.

**Status: do not restart this without budget for the bisection below.** The gain being chased is
78.45 s vs 85.64 s on t2va and 125.33 s vs 134.29 s on ref2va (§3.2.1), i.e. **7–9 % on top of
SageAttention**, and the last unexplored axis costs at least three arms to bisect. Sage is already
committed and working; this is the marginal lever, not the main one.

### What the failure looks like

Both arms rendered a full-length, correctly-shaped mp4 with no error in the log, at **12.45–12.51
Mbps against sage's 0.86–0.99 Mbps** — two independent arms within 0.5 % of each other. High bitrate
with valid structure is what an h264 encoder produces from noise, and it is why this cannot be caught
by anything except watching the video. Any future arm must be watched, not timed.

### The one hypothesis worth testing: tensor parallelism (G7.md §3.2.2)

`grep -- "tp-size\|tp_size\|TP=\|张量并行"` over all 72 KB of the `minimax_h3_g7e` README returns
**zero hits**. Every arm there is TP=1 on a 96 GB card, multi-card via Ulysses only. Our 32 GB cards
force TP=4. Every other axis of their delivered recipe matches ours exactly. And
`_copy_grouped_qkv_tp_shard` (`models/dits/minimax_h3.py:319`) demonstrably **refuses a packed 4-bit
weight** — twice, on `packed_dim` and on `dtype not in (bfloat16, float8_e4m3fn)` — so NVFP4 takes a
different shard-extraction path from the bf16 and online-fp8 arms that work.

### Bisection, ranked by information per GPU-minute

1. **Quantize everything except `qkv_proj`** — 156 layers instead of 208, one `QUANT_RE` edit,
   ~2 min CPU per partition, no new serve topology. Clean render ⇒ the defect is the qkv row/scale
   path under TP. Still noise ⇒ it is the FP4 GEMM or the row-parallel `out_proj` shard.
2. **NVFP4 at TP=1 with `--dit-layerwise-offload`**, however slow. Speed is irrelevant here; it is
   the only way to reach TP=1 on a 32 GB card, and it converts "TP is the suspect" into "TP is the
   cause" in a single arm.
3. **TP=2 × U=4** — only after 1 or 2 says the low-TP path is sound. See the TP=2 note below for why
   this needs `--minimax-h3-adaln-online`, and why the flag does *not* do what §3.2.2 first claimed.

### Things that must not be re-derived, and one that must not be redone

- **Do NOT apply either g7e source patch.** `patch_h3_qkv_scale_reorder.py` and
  `patch_nvfp4_tma_scale_layout.py` are both upstream on this image (#35739, #35740). Applying the
  qkv one on top reorders the rows **twice**, which is exactly as wrong as not reordering them at
  all, and just as silent. g7e verified the retirement criterion themselves: `FP4_UPSTREAM=1` with no
  patches and none of the three `H3_FP4_*` env vars produced **md5-identical** output
  (`174dcfd6…`). So "no patches, no env vars" is correct here and the missing env-var row in
  §3.2.2's table is not a discrepancy.
- **Do NOT pass `--quantization modelopt_fp4`.** The checkpoint carries its own
  `_quantization_metadata` header; the flag builds a second, empty config that wins and serves an
  unquantized-but-declared-quantized model. `QUANT=` (empty) is what keeps it off the command line.
- **AdaLN must stay bf16 in the file.** g7e casts the 50 `adaln_proj` weights to bare fp8; this image
  refuses that in two independent places (`quantization_utils.py:160` and `:206`/`:409`). `quant.sh`
  switches it off by monkeypatching `FP8_RE` to a never-matching pattern, leaving their file
  byte-identical. This is free in speed — those weights are 40 % of the DiT and 0 % of its FLOPs.
- **The self-check line, per partition:**
  `wrote /data/h3/nvfp4_<v>.safetensors: 951 tensors (q=208 fp8=0 copy=327) worst rel=0.0951`.
  0.094 ± 0.002 is the group-16 round-to-nearest-e2m1 floor. Materially higher means the packing or
  the scales are wrong; materially lower is impossible and means the check compared the wrong tensor.
- **Suspects already cleared, do not re-investigate:** the activation scale (`mg/modelopt_quant.py:671`
  sets `missing_param_init="ones"` and #35740 preserves it deliberately); the TMA scale reshape
  (unconditional after the trtllm early return at `:730`); `checkpoint_uses_native_qkv_layout`
  (defaults `False`, so the reorder *is* installed); the header flags
  (`packed_qkv=False, comfy_quant=False, scale_layout=linear, swap_nibbles=False` match the writer).
- **The two 37.5 GB `nvfp4_{fl2va,ref2va}.safetensors` files still exist** on the `h3-serve` pod's
  hostPath (`/data/h3/`, 199 GB HF cache alongside them, 6.4 T free). If a future session finds them
  missing, check the **kube context** before concluding the disk was recycled — see README, "Access".
  Rebuilding them is `quant.sh`, ~2 min CPU per partition, once the snapshot is on disk.

## TP=2 — it runs, it is 7.5 % slower, and it is a memory lever

**Answered, not parked** — see G7.md **§3.1.1**. `--dit-layerwise-offload` at TP=2 × U=4 (bf16, *no*
`--minimax-h3-adaln-online`; the two streaming loaders are incompatible) renders ref2va in
**144.38 s at 11 428 MB peak**, against fp8 TP=4 × U=2's 134.27 s at 27 690 MB. So it is not a speed
option, but it is the configuration with 20 GB of headroom per card, which is the one to reach for if
a longer clip or a larger reference image ever runs out of room at TP=4.

What remains open is only the *fast* low-TP route. Short form of why: **weights are the
problem and Ulysses cannot help, because Ulysses does not shard weights — only TP does.** Activation
residency is ~11 GiB per card at both TP=4 × U=2 and TP=2 × U=4, because the two shardings multiply
to the same 1/8, so halving TP buys nothing on the activation side and costs 15.9 GB/card of weights.
Even if it fit it would not win: bf16 (the only precision `--minimax-h3-adaln-online` accepts) costs
+19.7 % per step against fp8, versus a 10–15 % topology gain from halving TP.

**Online fp8 at TP=2 is closed, measured, not argued** (arm T4, G7.md §3.1.1). `--quantization fp8`
*plus* `--dit-layerwise-offload` at TP=2 × U=4 — both flags confirmed active in the server args — dies
at 29.47 GiB allocated inside `fsdp_load.py:load_model_from_full_model_state_dict`, while the identical
bf16 arm peaks at 11 428 MB. The quantized load path materialises parameters on device to attach weight
scales, so layerwise offload only streams what the unquantized path leaves streamable. Do not retry
this with a different offload flag combination; the failure is in the loader, not in residency.

## Offline fp8 checkpoint at TP=2 — the one unexplored configuration that could beat 134.27 s

**Not started.** 15.5 GB/card *and* fp8's 4.21 s/step, i.e. TP=2's memory headroom with no precision
penalty and no per-step streaming. `quantization_utils.py:409` accepts `["float8_e4m3fn"]` as a format
set, and `--transformer-weights-path` already serves an offline-quantized DiT (that is how NVFP4 loads),
so the missing piece is an exporter alongside `quant.sh` — same walk over the two partitions, `torch.
float8_e4m3fn` per-tensor or per-channel instead of the group-16 e2m1 packing, writing the same
`_quantization_metadata` header shape the loader reads at `:131`. Two cautions carried over from the
NVFP4 work: **AdaLN must stay bf16 in the file** (same two refusal sites), and the first render must be
**watched**, because a wrong scale layout renders cleanly and silently at the right bitrate.
