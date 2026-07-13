#!/usr/bin/env python3
"""Bewijs dat we MLC's q4f16_1-quantisatie bit-exact kunnen reproduceren.

Grondwaarheid: mlc-ai/phi-2-q4f16_1-MLC (officiële conversie) vs
microsoft/phi-2 (fp16-origineel). We fetchen alleen byte-ranges — geen
grote downloads.
"""
import json
import struct
import urllib.request

import numpy as np

MLC = "https://huggingface.co/mlc-ai/phi-2-q4f16_1-MLC/resolve/main/"
PHI = "https://huggingface.co/microsoft/phi-2/resolve/main/"


def fetch_range(url, start, length):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    assert len(data) == length, f"{len(data)} != {length}"
    return data


def st_header(url):
    """safetensors-header (JSON) van een shard, plus data-offset."""
    raw = fetch_range(url, 0, 8)
    (hlen,) = struct.unpack("<Q", raw)
    hdr = json.loads(fetch_range(url, 8, hlen))
    return hdr, 8 + hlen


def st_tensor(url, hdr, data0, name):
    info = hdr[name]
    start, end = info["data_offsets"]
    raw = fetch_range(url, data0 + start, end - start)
    dt = {"F16": np.float16, "F32": np.float32}[info["dtype"]]
    return np.frombuffer(raw, dtype=dt).reshape(info["shape"])


# ---- officiële MLC-records vinden ----
cache = json.load(open("ndarray-cache-phi2.json"))
loc = {}
for shard in cache["records"]:
    for r in shard["records"]:
        loc[r["name"]] = (shard["dataPath"], r)


def mlc_tensor(name):
    path, r = loc[name]
    dt = {"uint32": np.uint32, "float16": np.float16, "float32": np.float32}[r["dtype"]]
    raw = fetch_range(MLC + path, r["byteOffset"], r["nbytes"])
    return np.frombuffer(raw, dtype=dt).reshape(r["shape"])


def dequant(qw, qs):
    """q4f16_1: 8 nibbles per uint32, (q - 7) * scale, groepen van 32."""
    n, kp = qw.shape
    shifts = np.arange(8, dtype=np.uint32) * 4
    nib = ((qw[:, :, None] >> shifts) & 0xF).astype(np.float32)  # [n, k/8, 8]
    nib = nib.reshape(n, kp * 8)
    scale = np.repeat(qs.astype(np.float32), 32, axis=1)[:, : kp * 8]
    return (nib - 7.0) * scale


def quant_mine(w, variant):
    """Kandidaat-reproducties van q4f16_1."""
    n, k = w.shape
    g = w.reshape(n, k // 32, 32).astype(np.float32)
    maxabs = np.abs(g).max(axis=2)
    if variant == "f32scale":
        scale = (maxabs / 7.0).astype(np.float16)
    elif variant == "f64scale":
        scale = (maxabs.astype(np.float64) / 7.0).astype(np.float16)
    s32 = scale.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.round(g / s32[:, :, None]) + 7.0
    q = np.nan_to_num(q, nan=7.0)
    q = np.clip(q, 0, 14).astype(np.uint32).reshape(n, k // 8, 8)
    shifts = np.arange(8, dtype=np.uint32) * 4
    packed = (q << shifts).sum(axis=2, dtype=np.uint64).astype(np.uint32)
    return packed, scale


print("== stap 1: out_proj laag 0 — quantisatie-formule ==")
qw = mlc_tensor("transformer.h.0.mixer.out_proj.q_weight")
qs = mlc_tensor("transformer.h.0.mixer.out_proj.q_scale")

idx_hdr, idx0 = None, None
index = json.loads(urllib.request.urlopen(PHI + "model.safetensors.index.json").read())
wmap = index["weight_map"]
shard_dense = wmap["model.layers.0.self_attn.dense.weight"]
hdr, d0 = st_header(PHI + shard_dense)
w = st_tensor(PHI + shard_dense, hdr, d0, "model.layers.0.self_attn.dense.weight")
print("origineel:", w.shape, w.dtype)

for variant in ("f32scale", "f64scale"):
    pw, ps = quant_mine(w, variant)
    m_w = (pw == qw).mean()
    m_s = (ps.view(np.uint16) == qs.view(np.uint16)).mean()
    print(f"variant {variant}: q_weight match {m_w:.6f} | q_scale match {m_s:.6f}")

print()
print("== stap 2: Wqkv laag 0 — concat-volgorde ==")
qw2 = mlc_tensor("transformer.h.0.mixer.Wqkv.q_weight")
qs2 = mlc_tensor("transformer.h.0.mixer.Wqkv.q_scale")
deq = dequant(qw2, qs2)

parts = {}
for p in ("q_proj", "k_proj", "v_proj"):
    nm = f"model.layers.0.self_attn.{p}.weight"
    sh = wmap[nm]
    hdr2, d02 = st_header(PHI + sh)
    parts[p] = st_tensor(PHI + sh, hdr2, d02, nm).astype(np.float32)

plain = np.concatenate([parts["q_proj"], parts["k_proj"], parts["v_proj"]], axis=0)
# interleaved per head: (nh, 3, hd)
nh, hd = 32, 80
inter = np.stack([parts["q_proj"].reshape(nh, hd, 2560),
                  parts["k_proj"].reshape(nh, hd, 2560),
                  parts["v_proj"].reshape(nh, hd, 2560)], axis=1).reshape(7680, 2560)

for naam, cand in (("plain [q;k;v]", plain), ("interleaved (nh,3,hd)", inter)):
    err = np.abs(deq - cand).mean()
    print(f"{naam}: gemiddelde dequant-afwijking {err:.6f}")

print()
print("== stap 3: bias-mapping Wqkv ==")
b_mlc = mlc_tensor("transformer.h.0.mixer.Wqkv.bias").astype(np.float32)
bparts = []
for p in ("q_proj", "k_proj", "v_proj"):
    nm = f"model.layers.0.self_attn.{p}.bias"
    sh = wmap[nm]
    hdr3, d03 = st_header(PHI + sh)
    bparts.append(st_tensor(PHI + sh, hdr3, d03, nm).astype(np.float32))
bp = np.concatenate(bparts)
bi = np.stack([x.reshape(nh, hd) for x in bparts], axis=1).reshape(7680)
print("plain bias match:", np.abs(b_mlc - bp).max(), "| interleaved:", np.abs(b_mlc - bi).max())
