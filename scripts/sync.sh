#!/usr/bin/env bash
# Push this repo's scripts and prompts to the h3-serve pod.
#
#   bash scripts/sync.sh
#
# THIS REPO IS SELF-CONTAINED, EXCEPT THE PROMPTS. Everything the arms need is committed here: the two
# server scripts and the request driver (`sglang_{base,ref2va}_arm.sh`, `sglang_case.py`, `_env.sh`),
# and g7e's NVFP4 quantizer. It used to reach into ../minimax_h3_h100 for the first group and
# into ../../Trn2/minimax_h3_g7e for the last, on the theory that sharing beats copying. In practice
# that made the documented commands unrunnable from a fresh clone of this repo -- you could read a
# Quick Start that said `bash /data/h3/sglang_ref2va_arm.sh` and then not find that file anywhere in
# the project. Self-contained wins.
#
# THE COST IS DRIFT, so this script checks for it rather than pretending it cannot happen: if the
# other checkouts are present it diffs the shared files and warns. It does not copy or auto-fix --
# the H100 route's version is allowed to diverge (different card, different memory maths), and which
# way a difference should flow is a judgement call.
#
# WHY base64 AND NOT scp. The GPU nodes cannot be SSH'd at all; the only way in is `ssh Jump` plus
# `kubectl exec` into the pod. `kubectl cp` works from the jump host but not from here, so a file
# makes two hops: scp to the jump host, kubectl cp into the pod. base64 through a single tar keeps
# that to one round trip and survives the exec's text-mode stream.
#
# EVERYTHING LANDS FLAT IN /data/h3. That is the hostPath mount to the node's NVMe RAID0, and it has
# to be: the pod's own writable layer is capped by ephemeral-storage and kubelet evicts the pod for
# exceeding it, which is a 24 GB nvfp4 file's worth of easy mistake. The scripts expect the flat
# layout (VDNROOT=/data/h3), not the repo's scripts/ subdirectory -- which is also why case/*.txt
# lands beside them as /data/h3/demo_ref2va.txt rather than in a case/ directory.
#
# THE PROMPTS ARE THE EXCEPTION AND IT IS DELIBERATE. `case_ir.txt` / `case_t2va_v2.txt` are a
# customer's shot description, so they are not committed and `.gitignore` blocks them (`case/case_*.txt`
# and `ref/`). Only the synthetic `case/demo_*.txt` pair is tracked. This script ships whatever it
# finds in case/ and ref/, tracked or not, so the working copy is the right place to keep the real
# files -- they reach the pod without ever reaching a commit. Every glob below is therefore allowed to
# come up short: `cp` failing on an empty ref/ is not an error worth aborting a sync for.
set -euo pipefail
G=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
POD=${POD:-h3-serve}

# Drift check, advisory only. SHARED is the set that exists in both routes by history.
SHARED=(_env.sh sglang_base_arm.sh sglang_ref2va_arm.sh sglang_case.py)
H=${H100:-$G/../minimax_h3_h100}
if [ -d "$H/scripts" ]; then
  for f in "${SHARED[@]}"; do
    [ -f "$H/scripts/$f" ] || continue
    cmp -s "$G/scripts/$f" "$H/scripts/$f" ||
      echo "DRIFT: scripts/$f differs from $H/scripts/$f -- decide which way it should flow" >&2
  done
fi
Q=${G7E:-$G/../../Trn2/minimax_h3_g7e}
if [ -f "$Q/scripts/nvfp4_quantize_transformer.py" ]; then
  cmp -s "$G/scripts/nvfp4_quantize_transformer.py" "$Q/scripts/nvfp4_quantize_transformer.py" ||
    echo "DRIFT: scripts/nvfp4_quantize_transformer.py differs from g7e's at $Q" >&2
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp "$G"/scripts/*.sh "$tmp"/
cp "$G"/scripts/*.py "$tmp"/
shopt -s nullglob
cases=("$G"/case/*.txt)
[ ${#cases[@]} -eq 0 ] || cp "${cases[@]}" "$tmp"/
# ref/ keeps its directory because REFDIR=/data/h3/ref, not the flat root. Untracked by design: the
# reference images are customer material like the prompts.
refs=("$G"/ref/*)
if [ ${#refs[@]} -ne 0 ]; then mkdir -p "$tmp"/ref && cp "${refs[@]}" "$tmp"/ref/; fi
shopt -u nullglob

tar czf "$tmp/sync.tgz" -C "$tmp" $(cd "$tmp" && ls | grep -v sync.tgz)
base64 < "$tmp/sync.tgz" > "$tmp/sync.b64"
scp -q "$tmp/sync.b64" Jump:/tmp/sync.b64
ssh Jump "export PATH=\$HOME/bin:\$PATH
  kubectl cp /tmp/sync.b64 $POD:/tmp/sync.b64
  kubectl exec $POD -- bash -lc 'base64 -d /tmp/sync.b64 > /tmp/sync.tgz &&
    tar xzf /tmp/sync.tgz -C /data/h3 && ls -l /data/h3/*.sh | wc -l'"
echo "synced to $POD:/data/h3"
