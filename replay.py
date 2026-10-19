"""Replay a captured agent request against a running Strata server, optionally reshaped.

    python replay.py request/request.json --port 8080
    python replay.py request/request.json --port 8080 --drop-tools
    python replay.py request/request.json --port 8080 --drop-tools --keep-msg 0 1 --system-chars 3400

The body is sent as it was captured (model, messages, tools, stream, max_completion_tokens), so the server
renders and tokenizes the same prompt the client did.  The script prints the HTTP status, whether the stream
carried an error, the finish reason and the usage, so a #879 answer is easy to tell from a normal one.

Options that reshape the body, for bisecting:
  --drop-tools            remove the 48 tool schemas
  --tool-limit N          keep only the first N tools
  --tools-scale N         repeat every tool N times with distinct names (a bigger tool block)
  --drop-msg I ...        remove these 0-based message indexes
  --keep-msg I ...        keep only these 0-based message indexes
  --system-chars N        truncate message 0 (the system text) to N characters
  --max-tokens N          replace max_completion_tokens
"""
import argparse
import copy
import json
import sys
import time
import urllib.error
import urllib.request

MODEL = "qwen3.8-flash-next-iq3_s"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def scale_tools(tools, factor):
    """Copy every tool `factor` times with a distinct name, so the tool block grows without inventing prose."""
    out = []
    for i in range(factor):
        for t in tools:
            u = copy.deepcopy(t)
            if i:
                fn = u.setdefault("function", {})
                fn["name"] = f"{fn.get('name')}_{i}"
            out.append(u)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--tools-scale", type=int, default=1)
    ap.add_argument("--tool-limit", type=int, default=0, help="keep only the first N tools (0: all)")
    ap.add_argument("--drop-msg", type=int, nargs="*", default=[], help="0-based message indexes to remove")
    ap.add_argument("--keep-msg", type=int, nargs="*", default=[], help="keep only these 0-based indexes")
    ap.add_argument("--system-chars", type=int, default=0, help="truncate the first message to N characters")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--drop-tools", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    body = load(a.capture)
    body["model"] = MODEL
    if a.drop_tools:
        body.pop("tools", None)
    elif a.tools_scale > 1:
        body["tools"] = scale_tools(body.get("tools") or [], a.tools_scale)
    if a.tool_limit and body.get("tools"):
        body["tools"] = body["tools"][:a.tool_limit]
    if a.drop_msg:
        body["messages"] = [m for i, m in enumerate(body.get("messages") or []) if i not in set(a.drop_msg)]
    if a.keep_msg:
        body["messages"] = [m for i, m in enumerate(body.get("messages") or []) if i in set(a.keep_msg)]
    if a.system_chars and body.get("messages"):
        first = body["messages"][0]
        if isinstance(first.get("content"), str):
            first["content"] = first["content"][:a.system_chars]
    if a.max_tokens:
        body.pop("max_completion_tokens", None)
        body.pop("max_tokens", None)
        body["max_tokens"] = a.max_tokens
    data = json.dumps(body, ensure_ascii=False).encode()
    print(f"-> tools={len(body.get('tools') or [])} msg={len(body.get('messages') or [])} bytes={len(data)}",
          flush=True)
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0, text, finish, error, usage = time.time(), [], None, None, None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            status = r.status
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except ValueError:
                    continue
                if "error" in ev:
                    error = ev["error"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    text.append(d.get("content") or "")
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                if ev.get("usage"):
                    usage = ev["usage"]
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}", flush=True)
        return 1
    secs = time.time() - t0
    out = "".join(text)
    print(f"status={status} {secs:.1f}s finish={finish} usage={usage} chars={len(out)}", flush=True)
    if error:
        print(f"ERROR: {json.dumps(error, ensure_ascii=False)[:400]}", flush=True)
    print("text: " + out.replace("\n", " ")[:300], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
