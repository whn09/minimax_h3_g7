# MiniMax-H3 on g7.48xlarge — making 8 × RTX PRO 4500 fast

This repo is the **speed** project for one machine: `g7.48xlarge`, eight RTX PRO 4500 Blackwell
cards, **32 GB each**, `sm_120`. It was split out of
[`minimax_h3_h100`](https://github.com/whn09/minimax_h3_h100) so that the H100/B300 work and the g7
work stop sharing one document tree; the split is at
`minimax_h3_h100@bb0b08b`, which is where `G7.md`, `scripts/g7_sweep.sh` and `scripts/sage.sh` came
from and where their history still is.

## What is here

| file | what it is |
|---|---|
| `G7.md` | the findings document. Read this first — it is the argument, not a changelog. |
| `scripts/g7_sweep.sh` | the topology/precision sweep: what fits on 32 GB and what it costs. |
| `scripts/sage.sh` | SageAttention 2 arms. **1.23× t2va / 1.32× ref2va** at slightly *lower* peak memory. |
| `scripts/quant.sh` | offline NVFP4 quantization of the two DiT partitions, ~10 min CPU each. Writes the *canonical* layout — feed it through `nvfp4_comfy_layout.py` before serving. |
| `scripts/fp8_quantize_transformer.py` | offline **fp8** conversion. The qkv reorder in it is load-bearing — read its header. |
| `scripts/fp8off.sh` | the offline-fp8 arms. The fastest **exact-math** configuration, and the sage quality gate (F6). |
| `scripts/nvfp4.sh` | the first NVFP4 arms. They render **noise** — kept as the failure signature; use `nvfp4c.sh`. |
| `scripts/nvfp4_comfy_layout.py` | rewrites an NVFP4 checkpoint into the layout the H3 loader *asserts*. Read its header: it is the diagnosis. |
| `scripts/nvfp4c.sh` | the fixed NVFP4 arms. **The fastest configuration on this box**, at TP=1 × U=8. |
| `scripts/sync.sh` | push the scripts to the pod. Also the one place the cross-repo dependency is written down. |

## The one thing to know before running anything

**32 GB per card is the whole story.** `--quantization fp8` is *online*: the loader lands the
65.65 GiB bf16 checkpoint on the cards and casts there, so pure Ulysses (which replicates the DiT on
every card) cannot load at all. **`--dit-layerwise-offload` does not rescue it either** — it makes
*bf16* load at TP=2 (11.4 GB peak, G7.md §3.1.1) but fp8 still OOMs at 29.47 GiB in the loader, because
the quantized path materialises parameters on device to attach weight scales. So every *online*-fp8 arm
here is `TP=4 × ULYSSES=2` — pinned by the loader, not by the collectives — and that constraint is also
why the g7e project's numbers are not directly comparable to ours: that machine has 96 GB cards and
every arm of theirs is TP=1.

**The way out is an offline checkpoint, and it removes the topology constraint entirely.** Pre-quantize
the DiT with `scripts/fp8_quantize_transformer.py` and there is no cast to land, so `TP=2 × ULYSSES=4`
loads and runs **1.12–1.15× faster than the online-fp8 TP=4 floor at ~40 % of the peak memory, and the
gain adds no approximation the baseline did not already have** (sage and fp8 weights are in both) —
119.43 s ref2va / 74.65 s t2va for a 5 s 768p clip at 25 steps, or 56.64 s / 38.66 s with Cache-DiT
stacked on. G7.md §3.1.2.

**The floor is one step further down: offline NVFP4 at `TP=1 × ULYSSES=8`.** Every all-reduce leaves the
linears, and on a box with no NVLink that is worth 1.23× over TP=4 — **101.34 s ref2va / 64.62 s t2va**
at 25 steps, or **48.46 s / 33.54 s** with Cache-DiT rdt 0.16, at ~11.5 GB peak. TP=1 needs
`--dit-layerwise-offload` (nothing else shards 37.5 GB onto a 32 GB card) and NVFP4 needs
`scripts/nvfp4_comfy_layout.py`, because **declaring a tensor `nvfp4` makes the H3 loader assert three
layout properties of the file rather than read them** — swizzled scales, swapped nibbles, native qkv
rows. Getting any one of them wrong renders noise with no error in the log, which is exactly what the
first NVFP4 arms did. G7.md §3.2.3. NVFP4 is a coarser approximation than fp8 (round-trip 0.095 vs
0.0265), so these renders need watching in a way the fp8 ones do not.

## Every measured configuration, in one place

Everything below is **one machine, one seed (42), 768p, 121 frames = 5.04 s, reference short edge
2048**. `inference` is what the request driver reports — DiT + VAE, excluding the 70–120 s server
start. `s/step` is `inference ÷ steps` and is the only column that compares across step counts.
**Sorted fastest first.** `G7.md` is where each row is argued; this table is only the index.

Legend for the axes:

- **precision** — `bf16` = stock checkpoint. `fp8 on` = `--quantization fp8`, cast on the card at load.
  `fp8 off` / `NVFP4 off` = pre-quantized file served through `--transformer-weights-path`
  (`fp8_quantize_transformer.py` / `quant.sh` + `nvfp4_comfy_layout.py`). Round-trip weight error:
  fp8 **0.0265**, NVFP4 **0.095**.
- **TP × U** — `--tp-size` × `--ulysses-degree`. Product is always 8 (all eight cards). TP shards
  weights *and* activations; Ulysses shards activations only. 56 attention heads, so U ∈ {1,2,4,8}.
- **attention** — `--attention-backend`. `sdpa` is the unapproximated path; `sage` is SageAttention 2
  built from source at `d1a57a5` (int8 QK, fp8 PV → an approximation). Text encoder and both VAEs are
  pinned to `torch_sdpa` on every sage row.
- **Cache-DiT** — per-request `residual_diff_threshold` / `max_cached_steps`, warmup 4 on all of them.
  Process-wide enable is `SGLANG_CACHE_DIT_ENABLED`; **`--cache-dit-config` is a diffusers-only flag and
  does nothing here**.
- **memory levers** — what matters is *how often* each one pays, not how much it frees. **`DiT⇄` is the
  only per-step one**: `--dit-layerwise-offload` streams the transformer block weights from host RAM on
  every step, which is why it is the lever that unlocks TP=1 and also the only one with a recurring
  bill (37.5 GB/card/step at TP=1 against 18.7 at TP=2). Everything else pays **once per request**:
  `te⇄` = `--layerwise-offload-components text_encoder`, and `→cpu` = `--text-encoder-cpu-offload` /
  `--image-encoder-cpu-offload` / `--vae-cpu-offload` — those components run once per clip anyway.
  `adaln` = `--minimax-h3-adaln-online` streams the 50 `adaln_proj` weights, is **bf16-only**, and
  **cannot be stacked with `DiT⇄` at TP=2** (two streaming loaders → T2 below).
- **≈** marks the approximations stacked in that row, so a row is only comparable to another with the
  same mark: **W** quantized weights, **A** approximate attention, **C** block caching, **D** distilled
  8-step LoRA. A row with more marks is not a faster version of a row with fewer — it is a different
  output.

### ref2va (the harder task: it also holds the reference tower at short edge 2048)

| arm | precision | TP × U | attention | Cache-DiT | memory levers | ≈ | steps | inference | s/step | peak/card |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| **N8** | **NVFP4 off** | **1 × 8** | sage | **0.16 / mc 3** | DiT⇄ te⇄ img/vae→cpu | WAC | 25 | **48.46 s** | **1.94** | 11 788 MB |
| F5 | fp8 off | 2 × 4 | sage | 0.16 / mc 3 | DiT⇄ te⇄ img/vae→cpu | WAC | 25 | 56.64 s | 2.27 | 11 828 MB |
| C2 | fp8 on | 4 × 2 | sage | 0.16 / mc 3 | — | WAC | 25 | 62.30 s | 2.49 | 28 110 MB |
| r_lora8_adaln | bf16 + ref2v LoRA | 4 × 2 | sdpa | — | adaln te→cpu | D | 8 | 63.26 s | 7.91 | 29 776 MB |
| rir8 | bf16 + ref2v LoRA | 4 × 2 | sdpa | — | adaln te→cpu | D | 8 | 63.74 s | 7.97 | 29 776 MB |
| F5 | fp8 off | 2 × 4 | sage | 0.10 / mc 2 | DiT⇄ te⇄ img/vae→cpu | WAC | 25 | 65.95 s | 2.64 | 11 848 MB |
| C3 | fp8 on | 4 × 2 | sage | 0.10 / mc 2 | — | WAC | 25 | 72.05 s | 2.88 | 29 150 MB |
| r_lora8_tp8 | bf16 + ref2v LoRA | 8 × 1 | sdpa | — | — | D | 8 | 74.93 s | 9.37 | 27 238 MB |
| **N5** | **NVFP4 off** | **1 × 8** | sage | — | DiT⇄ te⇄ img/vae→cpu | WA | 25 | **101.34 s** | **4.05** | 11 588 MB |
| N6 | NVFP4 off | 2 × 4 | sage | — | DiT⇄ te⇄ img/vae→cpu | WA | 25 | 106.67 s | 4.27 | 11 548 MB |
| **F3** | **fp8 off** | **2 × 4** | sage | — | DiT⇄ te⇄ img/vae→cpu | WA | 25 | **119.43 s** | **4.78** | 11 428 MB |
| N4 *(control)* | NVFP4 off | 4 × 2 | sage | — | — | WA | 25 | 125.16 s | 5.01 | 28 142 MB |
| C1 | fp8 on | 4 × 2 | sage | 0.04 / mc 1 | — | WAC | 25 | 125.51 s | 5.02 | 28 310 MB |
| C0 / sage_ref2va | fp8 on | 4 × 2 | sage | — | — | WA | 25 | 134.27 s | 5.37 | 27 690 MB |
| F1 *(control)* | fp8 off | 4 × 2 | sage | — | — | WA | 25 | 134.53 s | 5.38 | 29 906 MB |
| T3 | bf16 | 2 × 4 | sage | — | DiT⇄ te⇄ img/vae→cpu | A | 25 | 144.38 s | 5.78 | 11 428 MB |
| **F6** | **fp8 off** | **2 × 4** | **sdpa** | — | DiT⇄ te⇄ img/vae→cpu | W | 25 | **163.99 s** | **6.56** | 11 488 MB |
| baseline | fp8 on | 4 × 2 | sdpa | — | — | W | 25 | 177.18 s | 7.09 | — |
| ~~nvfp4_ref2va~~ | NVFP4 off, **layout wrong** | 4 × 2 | sage | — | te⇄ | — | 25 | ~~125.33 s~~ | — | 20 968 MB |

Two rows above need a caveat rather than a footnote marker. The **`baseline` sdpa row** is §2.1's
`torch_sdpa` measurement on the 3337-char IR prompt; its peak was not recorded separately from the sage
arm's, and §1's `r_base` (an earlier, shorter prompt) measured **174.43 s / 27 872 MB** — treat anything
inside ~3 s here as prompt length, not topology. **T3 is bf16 *with* sage**, which is why it is only
1.08× slower than F6's fp8-without-sage despite carrying full-precision weights: the two rows differ on
both axes and are not a precision comparison. The clean sage measurement is F6 against F3.

### t2va

| arm | precision | TP × U | attention | Cache-DiT | memory levers | ≈ | steps | inference | s/step | peak/card |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| **N9** | **NVFP4 off** | **1 × 8** | sage | **0.16 / mc 3** | DiT⇄ te⇄ img/vae→cpu | WAC | 25 | **33.54 s** | **1.34** | 11 168 MB |
| G1 | fp8 off | 2 × 4 | sage | 0.16 / mc 3 | DiT⇄ te⇄ img/vae→cpu | WAC | 25 | 38.66 s | 1.55 | 11 248 MB |
| t2va8 | bf16 + fl2v LoRA | 4 × 2 | sdpa | — | adaln te→cpu | D | 8 | 39.42 s | 4.93 | 27 414 MB |
| D2 | fp8 on | 4 × 2 | sage | 0.16 / mc 3 | — | WAC | 25 | 43.04 s | 1.72 | 27 464 MB |
| **N7** | **NVFP4 off** | **1 × 8** | sage | — | DiT⇄ te⇄ img/vae→cpu | WA | 25 | **64.62 s** | **2.58** | 11 128 MB |
| **G1** | **fp8 off** | **2 × 4** | sage | — | DiT⇄ te⇄ img/vae→cpu | WA | 25 | **74.65 s** | **2.99** | 10 968 MB |
| sage_t2va / D0 | fp8 on | 4 × 2 | sage | — | — | WA | 25 | 85.85 / 86.03 s | 3.43 | 27 088 MB |
| D1 | fp8 on | 4 × 2 | sage | **0.04 / mc 1** | — | WA | 25 | 87.02 s | 3.48 | 27 404 MB |
| cudnn | fp8 on | 4 × 2 | `torch_cudnn_sdpa` | — | — | W | 25 | 104.35 s | 4.17 | 27 770 MB |
| sdpa | fp8 on | 4 × 2 | sdpa | — | — | W | 25 | 105.37 s | 4.21 | 27 770 MB |
| cachedit | fp8 on | 4 × 2 | sdpa | `--cache-dit-config` | — | W | 25 | 105.40 s | 4.22 | 27 770 MB |
| fa | fp8 on | 4 × 2 | `fa` → sdpa | — | — | W | 25 | 105.68 s | 4.23 | 27 770 MB |
| bf16_tp4u2c | bf16 | 4 × 2 | sdpa | — | adaln te/img/vae→cpu | — | 25 | 125.90 s | 5.04 | 22 562 MB |
| compile | fp8 on + `--enable-torch-compile` | 4 × 2 | sdpa | — | — | W | 25 | 130.53 s | 5.22 | 25 568 MB |
| tp8u1 | fp8 on | 8 × 1 | sdpa | — | — | W | 25 | 142.77 s | 5.71 | 22 024 MB |
| ~~nvfp4_t2va~~ | NVFP4 off, **layout wrong** | 4 × 2 | sage | — | — | — | 25 | ~~78.45 s~~ | — | 27 178 MB |

**The t2va rows are not all on the same prompt**, which is the one place this table is less matched than
the ref2va one. Three were used: the original sweep prompt (`cudnn`…`tp8u1`), the 1036-char IR prompt
(`sage_t2va`), and `case_t2va_v2.txt @wide` (`D*`, `G1`, `N7`, `N9` — the one
[`PROMPT_IR.md`](https://github.com/whn09/minimax_h3_h100/blob/main/PROMPT_IR.md) settled on, and the
one to use). It is worth ~0.2 % — 85.85 s on the IR prompt against
86.03 s on `@wide` at an otherwise identical configuration — so differences under ~1 s across these rows
are prompt length, not the knob in the column you are reading.

### What does not run at all, and why

| config | outcome |
|---|---|
| `fp8 on`, TP=2 × U=4 | **load OOM.** Online fp8 lands the 65.65 GiB bf16 checkpoint on the cards and casts there. |
| `fp8 on` + `DiT⇄`, TP=2 × U=4 (T4) | **load OOM**, 29.47 of 31.37 GiB. The quantized load path materialises parameters on device to attach weight scales, so streaming cannot help. |
| `fp8 off`, TP=2 × U=4, no `DiT⇄` (F2) | **forward OOM, 64 MB short.** Weights fit at TP=2 once pre-quantized; ~1 GB of activation headroom is what is missing. |
| F2 + `te⇄` only (F4) | **forward OOM, 118 MB short.** A once-per-request offload does not find that last gigabyte; only per-step DiT streaming does. |
| `bf16` + `adaln` + all offloads, TP=2 × U=4 (T1) | forward OOM, 30.55 of 31.37 GiB. |
| T1 + `DiT⇄` (T2) | **software wall**, not memory: `RuntimeError: No backend type associated with device type cpu`. Two streaming loaders cannot be stacked. |
| `bf16`, TP=1 × U=8 | load OOM. Ulysses does not shard weights. |
| `NVFP4 off`, TP=4 × U=2, no offload | load OOM, 30.59 of 31.37 GiB. |
| `fp8 on` + `adaln` | `--minimax-h3-adaln-online` is bf16-only. |
| `--attention-backend fa2` | fails on sm_120. |
| `--attention-backend video_sparse_attn_h3` | hard-gated off sm_120, and it does not override `forward_varlen`, which H3's packed-varlen DiT requires. |
| per-request `quality: "high"` | rejected — the audited Cache-DiT preset is gated to 4×H200 fl2va. |
| `bf16` + LoRA, TP=4 × U=2, without `adaln` | warmup OOM. bf16 + adapter is 15.4 GB/card of weights. |

### Three things this table says that are easy to miss

1. **Attention backend is a bigger lever than topology.** `sdpa → sage` is 1.32× at TP=4 and **1.37× at
   TP=2** (F6 vs F3), while the whole `TP=4 → TP=1` move is 1.23×. `fa`, `fa2` and `cudnn` are all
   within 1 % of `sdpa` or broken — on this card there is exactly one attention lever.
2. **The Cache-DiT dial is steep and the vendor's audited preset is worthless here.** rdt 0.04 buys 7 %
   on ref2va and is **byte-identical** on t2va (the cache never fires — `md5 4ddbf2e7…` for both D0 and
   D1). 0.10 buys 1.86×, 0.16 buys 2.16×. Usable range is 0.10–0.16, and *identical md5 means nothing
   happened*, which timing alone cannot distinguish from "the cache is on and cheap".
3. **Peak memory and speed stopped trading against each other.** The fastest rows are also the
   *smallest* (~11 GB against ~28 GB), because pre-quantized weights plus DiT streaming attack residency
   directly while removing the PCIe all-reduces that cost time. There is ~20 GB free per card at the
   floor, which is the headroom for a longer clip or a bigger reference image.

## This repo is not self-contained, on purpose

The serving drivers are **shared with the H100 route and live in the other repo**:

    ../minimax_h3_h100/scripts/_env.sh                 CUDA_HOME discovery, NCCL, ffmpeg gate
    ../minimax_h3_h100/scripts/sglang_base_arm.sh       the t2va/fl2va server
    ../minimax_h3_h100/scripts/sglang_ref2va_arm.sh     the ref2va server
    ../minimax_h3_h100/scripts/sglang_case.py           the request driver
    ../minimax_h3_h100/case/                            the customer's prompts

They are shared rather than copied because they are genuinely the same code driving two machines,
and a copy would drift silently — the arms scripts already carry the g7 reasoning in their comments
(`sglang_base_arm.sh:56-61` is about this card's 32 GB). The cost is that `scripts/sync.sh` needs
both checkouts side by side. Prompt-engineering material (`PROMPT_IR.md`, `aud.sh`, `r2.sh`,
`v2.sh`, `case_*_v2.txt`) also stays there: those experiments ran on this machine but they are about
the prompt, not the hardware.

## Access

Through the jump host, per `../../riv2026/aim345-selfrun/ACCESS.md` — read it, it has real
prohibitions in it. GPU nodes cannot be SSH'd; everything is `ssh Jump` + `kubectl exec` into the
`h3-serve` pod, with renders landing on the hostPath mount `/data/h3` (the pod's own writable layer
is capped by `ephemeral-storage` and kubelet will evict the pod for exceeding it).

**Check the kube context first, every time.** The jump host has carried two, and `h3-serve` lives in
**`aim345-full`** (`arn:aws:eks:eu-south-2:579019700964:cluster/aim345-full`), not in `qual`:

    ssh Jump 'export PATH=$HOME/bin:$PATH
      kubectl config use-context arn:aws:eks:eu-south-2:579019700964:cluster/aim345-full'

Under the wrong context `kubectl exec h3-serve` answers `Error from server (NotFound): pods
"h3-serve" not found`, which reads exactly like the node having been recycled and the ~200 GB model
cache having gone with it. It cost an hour of planning a rebuild that was not needed. `kubectl
config current-context` is the first command of any session; if the API endpoint fails to resolve at
all, that cluster is deleted, which is what happened to `qual`.

**The machine bills ~$16/hour and the GPUs bill while idle.** Say so when a run is finished.
