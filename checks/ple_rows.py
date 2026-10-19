"""Read the PLE n-gram table rows for a token bigram and check them for extreme values.

The engine's PLE row is IQ4_NL (90 bytes per 160-value row); a corrupted row (an infinite fp16 scale, or values
far outside the table's usual range) is one way a prompt can push a layer's activations out of range.

    python ple_rows.py <ple.gguf> [token prev1 prev2]
"""
import struct
import sys

PLE_N_HEADS, HEADS_PER_NGRAM, NGRAM_SIZE = 16, 8, 3
EOS, NULL = 248044, -1
MULT = [23703573157769, 20109073645365, 8052911324071]
VOCAB = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
         20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]
OFFSET = [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
          160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]
IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def ngram_rows(tok, prev1, prev2):
    ctx = [tok, prev1 if prev1 is not None else NULL, prev2 if prev2 is not None else NULL]
    out = []
    for n in (2, 3):
        mixed = (ctx[0] * MULT[0]) & 0xFFFFFFFFFFFFFFFF
        for j in range(1, n):
            t = ctx[j]
            if t < 0:
                t = EOS
            mixed ^= (t * MULT[j]) & 0xFFFFFFFFFFFFFFFF
        for g in range(HEADS_PER_NGRAM):
            h = (n - 2) * HEADS_PER_NGRAM + g
            out.append(mixed % VOCAB[h] + OFFSET[h])
    return out


def f16(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def dequant(row):
    out = []
    for b in range(5):
        blk = row[b * 18:(b + 1) * 18]
        d = f16(struct.unpack("<H", blk[:2])[0])
        qs = blk[2:]
        for j in range(16):
            out.append(d * IQ4NL[qs[j] & 0x0F])
            out.append(d * IQ4NL[qs[j] >> 4])
    return out


class Gguf:
    def __init__(self, path):
        self.f = open(path, "rb")
        magic, version, n_tensors, n_kv = self._unpack("<IIQQ")
        assert magic == 0x46554747, magic
        self.alignment = 32
        self.kv = {}
        for _ in range(n_kv):
            key = self._string()
            typ = self._unpack("<I")[0]
            self.kv[key] = self._value(typ)
        if isinstance(self.kv.get("general.alignment"), int):
            self.alignment = self.kv["general.alignment"]
        self.tensors = []
        for _ in range(n_tensors):
            name = self._string()
            n_dims = self._unpack("<I")[0]
            dims = self._unpack("<" + "Q" * n_dims)
            ttype = self._unpack("<I")[0]
            off = self._unpack("<Q")[0]
            self.tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
        pos = self.f.tell()
        self.data = (pos + self.alignment - 1) // self.alignment * self.alignment

    def _unpack(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack(fmt, self.f.read(n))

    def _string(self):
        n = self._unpack("<Q")[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _value(self, typ):
        if typ in (0, 1, 7, 2, 3, 4, 5):
            fmt = {0: "B", 1: "b", 7: "?", 2: "H", 3: "h", 4: "I", 5: "i"}[typ]
            return self._unpack("<" + fmt)[0]
        if typ == 6:
            return self._unpack("<f")[0]
        if typ in (10, 11):
            return self._unpack("<" + ("Q" if typ == 10 else "q"))[0]
        if typ == 12:
            return self._unpack("<d")[0]
        if typ == 8:
            return self._string()
        if typ == 9:
            etyp, count = self._unpack("<IQ")
            vals = [self._value(etyp) for _ in range(count)] if etyp != 8 else [self._string() for _ in range(count)]
            return vals
        raise ValueError(f"unknown gguf value type {typ}")

    def read_rows(self, tensor, rows, row_bytes, stride_rows):
        """Read `rows` (row indices into a table of row width `row_bytes`), assuming row-major contiguous rows."""
        out = {}
        for r in rows:
            self.f.seek(self.data + tensor["offset"] + r * row_bytes)
            out[r] = self.f.read(row_bytes)
        return out


if __name__ == "__main__":
    path = sys.argv[1]
    g = Gguf(path)
    if len(sys.argv) > 2 and sys.argv[2] == "--scan":
        # sample rows across the table and report the largest |value| and any non-finite scale
        import random
        t = g.tensors[0]
        rows_total = t["dims"][1]
        random.seed(1)
        worst, bad = [], 0
        sample = [random.randrange(rows_total) for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 5000)]
        for r in sample:
            row = g.read_rows(t, [r], 90, None)[r]
            vals = dequant(row)
            mx = max(abs(v) for v in vals)
            if any(not (v == v) or abs(v) == float("inf") for v in vals):
                bad += 1
            worst.append((mx, r))
        worst.sort(reverse=True)
        print(f"sampled {len(sample)} rows of {rows_total}: non-finite rows {bad}")
        print("largest |value| rows:", worst[:10])
        allv = sorted(w[0] for w in worst)
        print("median row max", allv[len(allv) // 2], "p99", allv[int(len(allv) * .99)], "max", allv[-1])
        raise SystemExit(0)
    if len(sys.argv) > 2:
        tok = int(sys.argv[2])
        p1 = int(sys.argv[3]) if len(sys.argv) > 3 else None
        p2 = int(sys.argv[4]) if len(sys.argv) > 4 else None
        t = g.tensors[0]
        rows = ngram_rows(tok, p1, p2)
        print("rows for", (tok, p1, p2), "->", rows)
        got = g.read_rows(t, rows, 90, None)
        for r in rows:
            vals = dequant(got[r])
            mx = max(abs(v) for v in vals)
            scales = [f16(struct.unpack("<H", got[r][b * 18:b * 18 + 2])[0]) for b in range(5)]
            print(f"row {r:>10}: max|v| {mx:.4g}  scales {scales}")
        raise SystemExit(0)
    print("kv:", {k: (v if not isinstance(v, list) else f"array[{len(v)}]") for k, v in g.kv.items()})
    for t in g.tensors:
        n = 1
        for d in t["dims"]:
            n *= d
        print("tensor", t["name"], "dims", t["dims"], "type", t["type"], "offset", t["offset"], "elements", n)
