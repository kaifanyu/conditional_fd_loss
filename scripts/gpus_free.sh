#!/usr/bin/env bash
# Free GPUs per node, most-free first.  Usage: scripts/gpus_free.sh [partition]
set -euo pipefail
PART="${1:-batch}"
printf '%-24s %-9s %-22s %s\n' NODE STATE TYPE 'USED/TOTAL  FREE'
sinfo -h -N -p "${PART}" -O "nodehost:24,statecompact:12,gres:44,gresused:44" | awk '
{
  node=$1; state=$2; tot=$3; used=$4; t=0; u=0
  if (match(tot,  /:[0-9]+\(/)) t = substr(tot,  RSTART+1, RLENGTH-2)
  if (match(used, /:[0-9]+\(/)) u = substr(used, RSTART+1, RLENGTH-2)
  type = tot; sub(/^gpu:/,"",type); sub(/:[0-9]+\(.*/,"",type)
  if (t > 0) printf "%-24s %-9s %-22s %2d/%-2d      %2d\n", node, state, type, u, t, t-u
}' | sort -k5 -rn
