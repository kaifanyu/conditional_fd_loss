#!/usr/bin/env bash
# Unpack the per-class ImageNet train tars into ImageFolder directories.
#
# ILSVRC2012_img_train.tar unpacks to 1000 tars named <wnid>.tar; each of those
# must become train/<wnid>/*.JPEG. This finishes that second step, in parallel.
#
# Resumable and non-destructive: a class whose directory already holds as many
# files as its tar has entries is skipped, and a partially extracted class is
# simply re-extracted over (tar overwrites). Nothing is deleted unless you ask.
#
#   bash scripts/extract_imagenet_train.sh                  # 16-way
#   J=32 bash scripts/extract_imagenet_train.sh             # more parallelism
#   DELETE_TARS=1 bash scripts/extract_imagenet_train.sh    # rm each tar once done
#
# DELETE_TARS=1 is recoverable: the master ILSVRC2012_img_train.tar is untouched.
set -euo pipefail

: "${DATA_PATH:=/mnt/projects/jg/kaifany/dataset/imagenet}"
: "${J:=16}"
: "${DELETE_TARS:=0}"

TRAIN_DIR="${DATA_PATH}/train"
[[ -d "${TRAIN_DIR}" ]] || { echo "ERROR: no such directory: ${TRAIN_DIR}" >&2; exit 2; }

mapfile -t TARS < <(find "${TRAIN_DIR}" -maxdepth 1 -name '*.tar' | sort)
echo "### ${#TARS[@]} class tars to consider in ${TRAIN_DIR}"
echo "### parallelism J=${J}, DELETE_TARS=${DELETE_TARS}"
(( ${#TARS[@]} > 0 )) || { echo "### nothing to do"; exit 0; }

export DELETE_TARS
printf '%s\0' "${TARS[@]}" | xargs -0 -P "${J}" -I{} bash -c '
    set -euo pipefail
    tarball="$1"
    dir="${tarball%.tar}"
    want=$(tar -tf "$tarball" | wc -l)
    have=0
    [[ -d "$dir" ]] && have=$(find "$dir" -maxdepth 1 -type f | wc -l)
    if (( have == want )); then
        echo "skip    $(basename "$dir")  ($have files, already complete)"
    else
        mkdir -p "$dir"
        tar -xf "$tarball" -C "$dir"
        got=$(find "$dir" -maxdepth 1 -type f | wc -l)
        if (( got != want )); then
            echo "FAILED  $(basename "$dir")  ($got != $want)" >&2
            exit 1
        fi
        echo "done    $(basename "$dir")  ($got files, was $have)"
    fi
    if [[ "${DELETE_TARS}" == "1" ]]; then rm -f "$tarball"; fi
' _ {}

echo "### finished; verifying layout"
