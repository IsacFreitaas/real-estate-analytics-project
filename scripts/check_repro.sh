#!/usr/bin/env bash
# Runs the model comparison pipeline twice and confirms both runs produce
# byte-identical CV/test results, verifying reproducibility (Issue #5).
set -euo pipefail

cd "$(dirname "$0")/.."

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

for i in 1 2; do
    python -m scripts.model_comparison > "$TMP_DIR/run_$i.log"
    cp outputs/model_comparison_cv.json "$TMP_DIR/model_comparison_cv_$i.json"
    cp outputs/final_test_evaluation.json "$TMP_DIR/final_test_evaluation_$i.json"
done

if diff "$TMP_DIR/model_comparison_cv_1.json" "$TMP_DIR/model_comparison_cv_2.json" \
    && diff "$TMP_DIR/final_test_evaluation_1.json" "$TMP_DIR/final_test_evaluation_2.json"; then
    echo "Reproducibility check passed: both runs produced identical results."
else
    echo "Reproducibility check FAILED: runs produced different results."
    exit 1
fi
