#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-diff}"
GPU="${GPU:-T4}"
ZIP="$(mktemp -u)-proj.zip"

up()   { colab new -s "$SESSION" --gpu "$GPU"; }
down() { colab stop -s "$SESSION"; }

run() {
    rm -f "$ZIP"
    zip -rq "$ZIP" src/diffusion scripts/train_model.py config.yaml pyproject.toml .env \
        -x "*/__pycache__/*"
    colab upload -s "$SESSION" "$ZIP" /content/proj.zip
    rm -f "$ZIP"
    colab exec -s "$SESSION" -f scripts/remote_run.py
}

case "${1:-}" in
    up)   up ;;
    run)  run ;;
    down) down ;;
    all)  up; trap down EXIT; run ;;   # provision, run, always release the VM
    *)    echo "usage: $0 {up|run|down|all}" >&2; exit 1 ;;
esac
