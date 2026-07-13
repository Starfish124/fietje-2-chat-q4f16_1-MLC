#!/usr/bin/env python3
"""Grid-test: divisie-methode x add-precisie x round-regel tegen officiële nibbles."""
import json
import struct
import urllib.request

import numpy as np

MLC = "https://huggingface.co/mlc-ai/phi-2-q4f16_1-MLC/resolve/main/"
PHI = "https://huggingface.co/microsoft/phi-2/resolve/main/"


def fetch_range(url, start, length):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def st_header(url):
    (hlen,) = struct.unpack("<Q", fetch_range(url, 0, 8))
    return json.loads(fetch_range(url, 8, hlen)), 8 + hlen


cache = json.load(open("ndarray-cache-phi2.json"))
loc = {}
for shard in cache["records"]:
    for r in shard["records"]:
        loc[r["name"]] = (shard["dataPath"], r)


def mlc_tensor(name):
    path, r = loc[name]
    dt = {"uint32": np.uint32, "float16": np.float16}[r["dtype"]]
    return np.frombuffer(fetch_range(MLC + path, r["byteOffset"], r["nbytes"]), dtype=dt).reshape(r["shape"])


qw = mlc_tensor("transformer.h.0.mixer.out_proj.q_weight")
qs = mlc_tensor("transformer.h.0.mixer.out_proj.q_scale")
index = json.loads(urllib.request.urlopen(PHI + "model.safetensors.index.json").read())
sh = index["weight_map"]["model.layers.0.self_attn.dense.weight"]
hdr, d0 = st_header(PHI + sh)
info = hdr["model.layers.0.self_attn.dense.weight"]
s, e = info["data_offsets"]
w = np.frombuffer(fetch_range(PHI + sh, d0 + s, e - s), dtype=np.float16).reshape(info["shape"])

n, k = w.shape
shifts = np.arange(8, dtype=np.uint32) * 4
nib_off = ((qw[:, :, None] >> shifts) & 0xF).astype(np.int32).reshape(n, k)
srep = np.repeat(qs, 32, axis=1)  # f16 official scales

divs = {}
with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
    divs["div_f16"] = w / srep                                    # f16
    divs["div_f32"] = w.astype(np.float32) / srep.astype(np.float32)
    divs["mul_recip_f16"] = w * (np.float16(1.0) / srep)          # f16 reciprocal
    divs["mul_recip_f32"] = (w.astype(np.float32) * (np.float32(1.0) / srep.astype(np.float32)))

def rond(t, methode):
    t = t.astype(np.float64)
    if methode == "banker":
        return np.round(t)
    if methode == "half_away":
        return np.sign(t) * np.floor(np.abs(t) + 0.5)
    raise ValueError

best = []
for dnaam, d in divs.items():
    for add in ("f16", "f32"):
        t7 = (d.astype(np.float16) + np.float16(7)) if add == "f16" else (d.astype(np.float32) + np.float32(7))
        for r in ("banker", "half_away"):
            q = rond(t7, r)
            q = np.nan_to_num(q, nan=7.0)
            q = np.clip(q, 0, 14).astype(np.int32)
            m = (q == nib_off).mean()
            best.append((m, f"{dnaam} | add {add} | {r}"))
        # ook: round op de div zelf, dan +7 (int)
        for r in ("banker", "half_away"):
            q = rond(d.astype(np.float32), r) + 7
            q = np.nan_to_num(q, nan=7.0)
            q = np.clip(q, 0, 14).astype(np.int32)
            m = (q == nib_off).mean()
            best.append((m, f"{dnaam} | round-dan-+7 | {r}"))

for m, naam in sorted(best, reverse=True)[:8]:
    print(f"{m:.7f}  {naam}")
