#!/usr/bin/env python3
"""Pad fietje-2-chat's vocab (50297) naar 51200, zodat de gewichten passen op
WebLLM's voorgecompileerde phi-2 runtime (die 51200 verwacht, net als de
originele microsoft/phi-2). Extra rijen zijn nul: logit 0 is verwaarloosbaar
naast echte logits, dus die tokens worden in de praktijk nooit gesampled.
"""
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

MODELDIR = Path(sys.argv[1] if len(sys.argv) > 1 else "model")
TARGET = 51200

cfg_path = MODELDIR / "config.json"
cfg = json.loads(cfg_path.read_text())
old = cfg["vocab_size"]
if old == TARGET:
    print("vocab is al", TARGET)
    sys.exit(0)

index_path = MODELDIR / "model.safetensors.index.json"
index = json.loads(index_path.read_text())
shards = sorted(set(index["weight_map"].values()))

total = 0
for shard in shards:
    path = MODELDIR / shard
    tensors = load_file(path)
    changed = False
    for name, t in list(tensors.items()):
        axes = [i for i, d in enumerate(t.shape) if d == old]
        if axes:
            pad = [(0, 0)] * t.ndim
            pad[axes[0]] = (0, TARGET - old)
            tensors[name] = np.pad(t, pad)
            print(f"gepadded {name}: {t.shape} -> {tensors[name].shape} ({t.dtype})")
            changed = True
    total += sum(int(t.nbytes) for t in tensors.values())
    if changed:
        save_file(tensors, path, metadata={"format": "pt"})

index.setdefault("metadata", {})["total_size"] = total
index_path.write_text(json.dumps(index, indent=2))
cfg["vocab_size"] = TARGET
cfg_path.write_text(json.dumps(cfg, indent=2))
print(f"klaar: vocab {old} -> {TARGET}, total_size {total}")
