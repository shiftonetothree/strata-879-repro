"""Send one small chat request and say whether the engine's logits came out non-finite (#879).

Each request is sent in a NEW conversation: a rotating single-token prefix is put in front of the system text,
so no two requests share a cached prompt prefix while their length stays the same (every character in MARKERS
is one token in this model's vocabulary, checked with the model's own tokenizer).  The reported
`prompt_tokens` shows that the lengths match.

    python probe.py --system-file prompt/system.txt
    python probe.py --system "x 传"
    python probe.py --system-file prompt/system.txt --system-chars 3399 --append " 传"

One caveat found while bisecting: a reducer this small only fires while the engine is warm (a cached prefix);
from a cold engine the same 58-token prompt passes.  The full request in request/request.json reproduces from
any state.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

MODEL = "qwen3.8-flash-next-iq3_s"
# one token each in this model's vocab; 传 is deliberately left out (it is one half of the reducer)
MARKERS = "转说视字幕的了是在有和就不人都一这们"
_used = 0


def next_marker():
    global _used
    _used += 1
    return MARKERS[_used % len(MARKERS)] + "\n"


def ask(system, user="你好", port=8080, max_tokens=8, timeout=600):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                       "max_tokens": max_tokens, "temperature": 0.7}, ensure_ascii=False).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        return "ok", (out.get("usage") or {}).get("prompt_tokens"), ""
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        return ("nonfinite" if "non-finite" in text else f"http{e.code}"), None, text[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", help="the system text itself")
    ap.add_argument("--system-file", help="a file with the system text")
    ap.add_argument("--system-chars", type=int, default=0, help="truncate that text to N characters")
    ap.add_argument("--append", default="", help="text to append after the (truncated) system text")
    ap.add_argument("--user", default="你好")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--repeat", type=int, default=1, help="send the same probe N times, each in a new conversation")
    a = ap.parse_args()
    if a.system_file:
        with open(a.system_file, encoding="utf-8") as f:
            text = f.read()
    elif a.system:
        text = a.system
    else:
        text = "hello"
    if a.system_chars:
        text = text[:a.system_chars]
    text += a.append
    for _ in range(a.repeat):
        status, tokens, extra = ask(next_marker() + text, user=a.user, port=a.port, max_tokens=a.max_tokens)
        print(f"{status:10s} prompt_tokens={tokens} chars={len(text)} {extra}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
