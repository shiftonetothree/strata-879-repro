"""Scan the token embedding (IQ4_XS) of the native GGUF: per-row largest |value| and any non-finite block scale.

A row whose block scale is infinite (or whose values are far outside the usual range) turns the residual stream
of that token into garbage - the kind of thing that makes a later layer's activation explode.

    python emb_scan.py <native.gguf> [rows...]
"""
import struct
import sys

import numpy as np

IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)
QK = 256
BLOCK = 136


class Gguf:
    def __init__(self, path):
        self.f = open(path, "rb")
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", self.f.read(24))
        assert magic == 0x46554747
        self.alignment = 32
        self.kv = {}
        for _ in range(n_kv):
            key = self._string()
            typ = struct.unpack("<I", self.f.read(4))[0]
            self.kv[key] = self._value(typ)
        if isinstance(self.kv.get("general.alignment"), int):
            self.alignment = self.kv["general.alignment"]
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._string()
            n_dims = struct.unpack("<I", self.f.read(4))[0]
            dims = struct.unpack("<" + "Q" * n_dims, self.f.read(8 * n_dims))
            ttype = struct.unpack("<I", self.f.read(4))[0]
            off = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors[name] = {"dims": dims, "type": ttype, "offset": off}
        self.data = (self.f.tell() + self.alignment - 1) // self.alignment * self.alignment

    def _string(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _value(self, typ):
        simple = {0: "B", 1: "b", 7: "?", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 10: "Q", 11: "q", 12: "d"}
        if typ in simple:
            fmt = simple[typ]
            return struct.unpack("<" + fmt, self.f.read(struct.calcsize(fmt)))[0]
        if typ == 8:
            return self._string()
        if typ == 9:
            etyp, count = struct.unpack("<IQ", self.f.read(12))
            return [self._value(etyp) for _ in range(count)]
        raise ValueError(f"gguf type {typ}")

    def read(self, name, size, offset=0):
        t = self.tensors[name]
        self.f.seek(self.data + t["offset"] + offset)
        return self.f.read(size)


def scan(path, rows_wanted):
    g = Gguf(path)
    t = g.tensors["token_embd.weight"]
    n_embd, n_vocab = t["dims"][0], t["dims"][1]
    row_bytes = n_embd // QK * BLOCK
    print(f"token_embd.weight dims {t['dims']} type {t['type']} row_bytes {row_bytes} total {row_bytes * n_vocab:,}")
    raw = g.read("token_embd.weight", row_bytes * n_vocab)
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n_vocab, n_embd // QK, BLOCK)
    d = a[:, :, :2].copy().view(np.float16).reshape(n_vocab, -1).astype(np.float32)      # (V, 10)
    scales_h = a[:, :, 2:4].copy().view(np.uint16).reshape(n_vocab, -1)                  # (V, 10)
    scales_l = a[:, :, 4:8].reshape(n_vocab, n_embd // QK, 4)                            # (V, 10, 4)
    ib = np.arange(8)
    sh = ((scales_h[:, :, None] >> (2 * ib)) & 3) << 4                                    # (V, 10, 8)
    sel = scales_l[:, :, ib // 2]                                                         # (V, 10, 8)
    ls = np.where(ib % 2 == 0, sel & 0x0F, (sel >> 4) & 0x0F)
    ls = (ls | sh).astype(np.float32) - 32.0                                             # (V, 10, 8)
    dl = d[:, :, None] * ls                                                              # (V, 10, 8)
    rowmax = (np.abs(dl).max(axis=(1, 2)) * 127.0)                                       # (V,)
    bad = ~np.isfinite(d).all(axis=1) | ~np.isfinite(dl).all(axis=(1, 2))
    print(f"rows with a non-finite scale: {int(bad.sum())}")
    order = np.argsort(-rowmax)
    print("largest rows:", [(int(r), float(rowmax[r])) for r in order[:8]])
    finite = rowmax[np.isfinite(rowmax)]
    print(f"median row max {np.median(finite):.4g}  p99 {np.percentile(finite, 99):.4g}  p99.9 "
          f"{np.percentile(finite, 99.9):.4g}  max {finite.max():.4g}")
    for r in rows_wanted:
        print(f"row {r}: max|v| {rowmax[r]:.6g}  d[0]={d[r, 0]:.6g}  d={d[r, :4]}")


if __name__ == "__main__":
    rows = [int(x) for x in sys.argv[2:]] or [96202, 96156, 220]
    scan(sys.argv[1], rows)
