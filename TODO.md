# Open and closed work on g7.48xlarge

**Nothing on this page is parked any more.** All four items below are answered; they are kept because
each one carries a fact that would otherwise have to be rediscovered, and two of them record a
conclusion this repo got wrong first.

## NVFP4 — **DONE and unparked 2026-09-18. It was a checkpoint-layout mismatch, and it is now the floor.**

**Status: nothing left to do.** Built and measured: `scripts/nvfp4_comfy_layout.py` +
`scripts/nvfp4c.sh`, results in **G7.md §3.2.3**. **101.34 s ref2va / 64.62 s t2va at TP=1 × U=8**
with sage and `--dit-layerwise-offload`, and **48.46 s / 33.54 s** with Cache-DiT rdt 0.16 on top —
2.77× / 2.57× over the online-fp8 TP=4 floor at ~11.5 GB peak. Both converted files are on the pod's
hostPath: `nvfp4c_{ref2va,fl2va}.safetensors`, 37 475 504 096 bytes each, alongside the two original
`nvfp4_*` files they were derived from.

### The cause, and the bisection that is retired unrun

Not tensor parallelism. `resolve_minimax_h3_checkpoint_quantization`
(`runtime/loader/minimax_h3_weights.py:61`, from `component_loaders/transformer_loader.py:315`) does
not *infer* the layout of an nvfp4-marked H3 checkpoint — it **hard-codes** three properties of it
(`:85-88`): `checkpoint_uses_native_qkv_layout = True`, `checkpoint_weight_scale_layout = "swizzled"`,
`swap_weight_nibbles = True`. Our file is the opposite on all three, so the DiT multiplied by numbers
unrelated to the model, and would have done so at TP=1 too. `CHECK=1` on
`nvfp4_comfy_layout.py` proved it for **zero GPU time**: cos **−0.000** against the bf16 weights
before the fix, cos 1.000 after. The three-arm bisection that used to be in this section is retired
**unrun** — its item 2 would have spent an arm confirming the wrong hypothesis, and item 1 would have
come back clean for the wrong reason.

**The carry-forward fact is the layout contract, not the diagnosis.** Declaring a layer `nvfp4` in an
H3 checkpoint *asserts* the ComfyUI-kitchen convention — swizzled scales, high nibble = even index,
qkv rows already `[q_all,k_all,v_all]`. The exporter therefore has to produce that, which is what
`nvfp4_comfy_layout.py` does with the model's own `_reorder_grouped_qkv_to_qkv` plus a scale swizzle
and a nibble swap. All three are pure permutations, so the converted file's round-trip error is
exactly the source's 0.095. If those files ever have to be rebuilt: `nvfp4c.sh`'s `C`/`CB` arms,
~3 min CPU each, and the script refuses to write unless it saw 208 quantized layers, 52 qkv reorders
and a verified sample against the bf16 source (`worst cos=0.995 rel=0.095`).

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
- **Suspects genuinely cleared:** the activation scale (`mg/modelopt_quant.py:671` sets
  `missing_param_init="ones"` and #35740 preserves it deliberately) and the TMA scale reshape
  (unconditional after the trtllm early return at `:730`).
- **Two suspects that were cleared *wrongly*, recorded so the mistake is not repeated.**
  `checkpoint_uses_native_qkv_layout` was read as "defaults `False` (`configs/base_config.py:38`), so
  the reorder *is* installed" — that is the **class default**, and `minimax_h3_weights.py:86` overrides
  it to `True` for every nvfp4 checkpoint. And `_copy_grouped_qkv_tp_shard` refusing packed 4-bit
  weights is **irrelevant**, because with the reorder never installed that function is unreachable for
  NVFP4. Also: the header flags `scale_layout=linear, swap_nibbles=False` *did* match the writer — the
  bug was that the reader does not read them.
- **What the failure looked like, for pattern-matching next time.** A full-length, correctly-shaped mp4,
  no error in the log, at **12.45–12.51 Mbps against sage's 0.86–0.99 Mbps** — high bitrate with valid
  structure is what h264 produces from noise. The fixed arms all land at 0.93–1.17 Mbps. That band is a
  structural check and not a quality gate: every approximate arm still has to be watched.
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

## Offline fp8 checkpoint at TP=2 — **DONE, and it is the new floor.** Not parked.

Built and measured: `scripts/fp8_quantize_transformer.py` + `scripts/fp8off.sh`, results in
**G7.md §3.1.2**. 119.43 s ref2va / 74.65 s t2va at TP=2 × U=4 (1.12× / 1.15× over the online-fp8 TP=4
floor, exact math, ~11 GB peak), and 56.64 s / 38.66 s with Cache-DiT rdt=0.16 on top. Both files exist
on the pod's hostPath: `fp8_{ref2va,fl2va}.safetensors`, 46 242 233 688 bytes each.

Nothing here is left to do. The one thing to carry forward if the files ever have to be rebuilt:
**declaring a layer `float8_e4m3fn` asserts the qkv rows are already native**, because
`ComfyFp8Config.checkpoint_uses_native_qkv_layout = True` makes minimax_h3.py skip its own reorder —
so the exporter must reorder, and it refuses to write unless it reordered exactly 52 and quantized
exactly 208. **NVFP4 asserts the same thing** (`minimax_h3_weights.py:86`) plus swizzled scales and
swapped nibbles — the two contracts agree on qkv and NVFP4 adds two more axes. The earlier version of
this note said the opposite, and that error is what parked NVFP4 for a week; see the NVFP4 section
above.
