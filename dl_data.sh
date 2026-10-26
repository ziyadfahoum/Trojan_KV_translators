#!/bin/bash
# download a ~400MB head slice of each split, drop the truncated last line
cd "$(dirname "$0")"
for f in reasoning_off reasoning_on; do
  curl -s -L -H "Authorization: Bearer $HF_TOKEN" -r 0-419430400 \
    "https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2/resolve/main/data/$f.jsonl" \
    -o data/${f}_slice.jsonl
  head -n -1 data/${f}_slice.jsonl > data/$f.jsonl && rm data/${f}_slice.jsonl
  echo "$f lines: $(wc -l < data/$f.jsonl)"
done
