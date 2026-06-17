#!/bin/bash
# Usage: ./exps/best.sh results/my_run

RUN_DIR=${1:?Usage: $0 <run_dir>}

best_ckpt=""
best_laps=-1

for ckpt in "$RUN_DIR"/checkpoint_*.zip; do
    echo "=== $ckpt ==="
    python3 exps/eval.py \
        --model "$ckpt" \
        --config config.yaml \
        --episodes 10 \
        --max-steps 20000

    # eval.py writes to results/eval_<model_name>_10ep.csv
    model_name=$(basename "$ckpt" .zip)
    csv="results/eval_${model_name}_10ep.csv"

    if [ ! -f "$csv" ]; then
        echo "[warn] CSV not found: $csv — skipping"
        continue
    fi

    laps=$(tail -n +2 "$csv" | awk -F',' '{sum += ($3 == "True")} END {print sum+0}')

    if [ "$laps" -gt "$best_laps" ]; then
        best_laps=$laps
        best_ckpt=$ckpt
    fi
done

if [ -z "$best_ckpt" ]; then
    echo "[error] No checkpoints evaluated successfully."
    exit 1
fi

echo ""
echo "=== Best checkpoint: $best_ckpt ($best_laps laps) ==="
cp "$best_ckpt" "$RUN_DIR/model.zip"
echo "=== Copied to $RUN_DIR/model.zip ==="
