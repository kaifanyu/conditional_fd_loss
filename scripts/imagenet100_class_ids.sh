#!/usr/bin/env bash
# Single source of truth for the 100-class scaling-probe subset.
#
# Source this; do not copy the list. The class GMM fit, the three FD judge
# references and the training run must all agree on the subset. A mismatch
# between the GMM fit and --train_class_ids is caught at launch by
# ClassGMMReference.validate_labels, but a mismatch in the *FD* references is
# not: it would silently score the generator against the wrong real
# distribution and read as a FID regression.
#
# Selection: a uniform stride-10 sample of the ImageNet label space, NOT a
# superset of the 20-class diagnostic set. The probe exists to predict what
# happens at 1000 classes, so its class-difficulty distribution has to match the
# 1000-class problem. The 20 diagnostic classes were hand-picked to be visually
# distinct ("tench, ostrich, macaw, flamingo, ..."), so extending them would
# bias the probe optimistic exactly where it needs to be honest -- a stride
# sample keeps the fine-grained confusions (dog breeds, snakes) that make 1000
# classes hard. Five of the diagnostic classes (0, 130, 340, 360, 920) fall on
# the stride and give a partial anchor to the pilot.
#
# Override with IMAGENET100_CLASS_IDS_OVERRIDE="0 1 2 ..." if you want a
# different subset; every consumer picks it up.

if [[ -n "${IMAGENET100_CLASS_IDS_OVERRIDE:-}" ]]; then
    read -r -a IMAGENET100_CLASS_IDS <<< "${IMAGENET100_CLASS_IDS_OVERRIDE}"
else
    mapfile -t IMAGENET100_CLASS_IDS < <(seq 0 10 990)
fi

if (( ${#IMAGENET100_CLASS_IDS[@]} != 100 )); then
    echo "ERROR: expected 100 class IDs, got ${#IMAGENET100_CLASS_IDS[@]}" >&2
    exit 2
fi

export IMAGENET100_CLASS_IDS
