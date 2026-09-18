#!/usr/bin/env bash
# Push the scripts to the h3-serve pod, from BOTH checkouts.
#
#   bash scripts/sync.sh
#
# WHY IT REACHES INTO ANOTHER REPO. The g7 arms are driven by code shared with the H100 route
# (sglang_case.py and the two arm scripts), which lives in minimax_h3_h100 and is not duplicated here
# -- see README.md. So this script needs both checkouts side by side and says so if they are not.
#
# WHY base64 AND NOT scp. The GPU nodes cannot be SSH'd at all; the only way in is `ssh Jump` plus
# `kubectl exec` into the pod. `kubectl cp` works from the jump host but not from here, so a file
# makes two hops: scp to the jump host, kubectl cp into the pod. base64 through a single tar keeps
# that to one round trip and survives the exec's text-mode stream.
#
# EVERYTHING LANDS FLAT IN /data/h3. That is the hostPath mount to the node's NVMe RAID0, and it has
# to be: the pod's own writable layer is capped by ephemeral-storage and kubelet evicts the pod for
# exceeding it, which is a 24 GB nvfp4 file's worth of easy mistake. The scripts expect the flat
# layout (VDNROOT=/data/h3), not the repo's scripts/ subdirectory.
set -euo pipefail
G=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
H=${H100:-$G/../minimax_h3_h100}
POD=${POD:-h3-serve}

[ -d "$H/scripts" ] || { echo "no minimax_h3_h100 checkout at $H -- set H100=<path>"; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp "$G"/scripts/*.sh "$tmp"/
# *.py too: fp8_quantize_transformer.py is ours (unlike the nvfp4 one below, which is g7e's).
cp "$G"/scripts/*.py "$tmp"/ 2>/dev/null || true
cp "$H"/scripts/_env.sh "$H"/scripts/sglang_base_arm.sh "$H"/scripts/sglang_ref2va_arm.sh \
   "$H"/scripts/sglang_case.py "$tmp"/
cp "$H"/case/*.txt "$tmp"/
# The quantizer itself is the g7e project's file, copied verbatim on purpose (quant.sh explains why).
Q=${G7E:-$G/../../Trn2/minimax_h3_g7e}
[ -f "$Q/scripts/nvfp4_quantize_transformer.py" ] && cp "$Q/scripts/nvfp4_quantize_transformer.py" "$tmp"/ \
  || echo "note: no g7e checkout at $Q; nvfp4_quantize_transformer.py not synced"

tar czf "$tmp/sync.tgz" -C "$tmp" $(cd "$tmp" && ls | grep -v sync.tgz)
base64 < "$tmp/sync.tgz" > "$tmp/sync.b64"
scp -q "$tmp/sync.b64" Jump:/tmp/sync.b64
ssh Jump "export PATH=\$HOME/bin:\$PATH
  kubectl cp /tmp/sync.b64 $POD:/tmp/sync.b64
  kubectl exec $POD -- bash -lc 'base64 -d /tmp/sync.b64 > /tmp/sync.tgz &&
    tar xzf /tmp/sync.tgz -C /data/h3 && ls -l /data/h3/*.sh | wc -l'"
echo "synced to $POD:/data/h3"
