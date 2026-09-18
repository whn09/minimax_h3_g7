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
| `scripts/quant.sh` | offline NVFP4 conversion of the two DiT partitions. Pure CPU, ~10 min each. |
| `scripts/nvfp4.sh` | NVFP4 weights **stacked on sage**. Weight quantization alone is a regression at 768p. |
| `scripts/sync.sh` | push the scripts to the pod. Also the one place the cross-repo dependency is written down. |

## The one thing to know before running anything

**32 GB per card is the whole story.** `--quantization fp8` is *online*: the loader lands the
65.65 GiB bf16 checkpoint on the cards and casts there, so pure Ulysses (which replicates the DiT on
every card) cannot load at all. Every arm here is therefore `TP=4 × ULYSSES=2`, and that constraint
is also why the g7e project's numbers are not directly comparable to ours — that machine has 96 GB
cards and every arm of theirs is TP=1.

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

**The machine bills ~$16/hour and the GPUs bill while idle.** Say so when a run is finished.
