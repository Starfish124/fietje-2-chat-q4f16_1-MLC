#!/usr/bin/env python3
"""Converteer BramVanroy/fietje-2-chat naar MLC q4f16_1 (WebLLM), zonder de
mlc_llm-toolchain: pure numpy, met de officiële mlc-ai/phi-2-q4f16_1-MLC als
byte-layout-template. De quantisatie is bit-exact geverifieerd tegen die
officiële conversie (zie diagnose_round.py):

  scale  = maxabs_per_groep32 * f16(1/7)          (f16-arithmetiek)
  q      = bankers_round( f16(w/scale) + f16(7) ) (add in f16!)  -> clip [0,14]
  pack   = 8 nibbles per uint32, LSB eerst

Vocab wordt gepadded 50297 -> 51200 (lay-out van originele phi-2; extra rijen
nul, worden nooit gesampled).
"""
import json
import struct
from pathlib import Path

import numpy as np

HIER = Path(__file__).parent
MODEL = HIER / "fietje-model"
OUT = HIER / "out"
OUT.mkdir(exist_ok=True)
TARGET_VOCAB = 51200

# ---------- fietje-tensors (lazy, bf16 -> f16) ----------
index = json.loads((MODEL / "model.safetensors.index.json").read_text())
handles = dict(index["weight_map"])


class Shard:
    """Handmatige safetensors-lezer (mmap) — safetensors-numpy kan geen bf16."""

    def __init__(self, pad):
        with open(pad, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(hlen))
        self.data0 = 8 + hlen
        self.mm = np.memmap(pad, dtype=np.uint8, mode="r")

    def tensor(self, naam):
        info = self.header[naam]
        s, e = info["data_offsets"]
        raw = self.mm[self.data0 + s:self.data0 + e]
        if info["dtype"] == "BF16":
            u16 = raw.view(np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            t = f32.astype(np.float16)  # round-to-nearest, zoals mlc's cast
        elif info["dtype"] == "F16":
            t = np.array(raw.view(np.float16))
        elif info["dtype"] == "F32":
            t = raw.view(np.float32).astype(np.float16)
        else:
            raise TypeError((naam, info["dtype"]))
        return t.reshape(info["shape"])


opens = {s: Shard(MODEL / s) for s in set(handles.values())}


def hf_tensor(naam):
    return opens[handles[naam]].tensor(naam)


def pad_vocab(t):
    if t.shape[0] == 50297:
        pad = [(0, TARGET_VOCAB - 50297)] + [(0, 0)] * (t.ndim - 1)
        return np.pad(t, pad)
    return t


def bron(naam_mlc):
    """Map een phi-msft-basisnaam naar de fietje-tensor(s)."""
    if naam_mlc == "transformer.embd":
        return pad_vocab(hf_tensor("model.embed_tokens.weight"))
    if naam_mlc.startswith("lm_head.linear"):
        soort = naam_mlc.rsplit(".", 1)[1]  # weight-achtig of bias
        return pad_vocab(hf_tensor(f"lm_head.{soort}"))
    if naam_mlc.startswith("lm_head.ln"):
        soort = naam_mlc.rsplit(".", 1)[1]
        return hf_tensor(f"model.final_layernorm.{soort}")
    if naam_mlc.startswith("transformer.h."):
        rest = naam_mlc[len("transformer.h."):]
        laag, sub = rest.split(".", 1)
        pfx = f"model.layers.{laag}"
        if sub.startswith("ln."):
            return hf_tensor(f"{pfx}.input_layernorm.{sub[3:]}")
        if sub.startswith("mixer.Wqkv."):
            soort = sub.rsplit(".", 1)[1]
            return np.concatenate([hf_tensor(f"{pfx}.self_attn.{p}.{soort}")
                                   for p in ("q_proj", "k_proj", "v_proj")], axis=0)
        if sub.startswith("mixer.out_proj."):
            soort = sub.rsplit(".", 1)[1]
            return hf_tensor(f"{pfx}.self_attn.dense.{soort}")
        if sub.startswith("mlp."):
            return hf_tensor(f"{pfx}.{sub}")
    raise KeyError(naam_mlc)


# ---------- bit-exacte q4f16_1 ----------
EEN_ZEVENDE = np.float16(1.0) / np.float16(7.0)
SHIFTS = np.arange(8, dtype=np.uint32) * 4


def quantize(w16):
    """w16 [n,k] f16 -> (q_weight [n,k/8] uint32, q_scale [n,k/32] f16)"""
    n, k = w16.shape
    assert k % 32 == 0
    g = w16.reshape(n, k // 32, 32)
    maxabs = np.abs(g).max(axis=2)                       # f16, exact
    scale = (maxabs * EEN_ZEVENDE).astype(np.float16)    # bit-exact formule
    s32 = np.repeat(scale.astype(np.float32), 32, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = w16.astype(np.float32) / s32
    t7 = (d.astype(np.float16) + np.float16(7))          # add in f16!
    q = np.round(t7.astype(np.float64))                  # banker's
    q = np.nan_to_num(q, nan=7.0)
    q = np.clip(q, 0, 14).astype(np.uint32).reshape(n, k // 8, 8)
    packed = np.zeros((n, k // 8), dtype=np.uint32)
    for j in range(8):
        packed |= q[:, :, j] << SHIFTS[j]
    return packed, scale


# ---------- shards schrijven volgens template ----------
cache = json.loads((HIER / "ndarray-cache-phi2.json").read_text())
quant_cache = {}

def waarde(rec):
    naam = rec["name"]
    if naam.endswith(".q_weight") or naam.endswith(".q_scale"):
        basis = naam.rsplit(".", 1)[0]
        if basis not in quant_cache:
            w = bron(basis if basis == "transformer.embd" else basis + ".weight")
            quant_cache[basis] = quantize(w)
            while len(quant_cache) > 3:  # geheugen beperken
                quant_cache.pop(next(iter(quant_cache)))
        qw, qs = quant_cache[basis]
        arr = qw if naam.endswith(".q_weight") else qs
    else:
        arr = bron(naam)
    verwacht = {"uint32": np.uint32, "float16": np.float16}[rec["dtype"]]
    arr = np.ascontiguousarray(arr.astype(verwacht, copy=False))
    assert list(arr.shape) == rec["shape"], (naam, arr.shape, rec["shape"])
    assert arr.nbytes == rec["nbytes"], (naam, arr.nbytes, rec["nbytes"])
    return arr


totaal = 0
for shard in cache["records"]:
    pad = OUT / shard["dataPath"]
    with open(pad, "wb") as f:
        for rec in shard["records"]:
            assert f.tell() == rec["byteOffset"], (rec["name"], f.tell(), rec["byteOffset"])
            f.write(waarde(rec).tobytes())
    grootte = pad.stat().st_size
    assert grootte == shard["nbytes"], (shard["dataPath"], grootte, shard["nbytes"])
    totaal += grootte
    print(f"{shard['dataPath']}: {grootte/1048576:.1f} MB OK")

print(f"klaar: {totaal/1048576:.0f} MB in {len(cache['records'])} shards")

# ndarray-cache.json is byte-voor-byte gelijk aan de template-lay-out
(OUT / "ndarray-cache.json").write_text(json.dumps(cache, indent=2))
print("ndarray-cache.json geschreven")
