"""Send the captured request to a running Strata server, exactly as the client sent it.

    python replay.py request/request.json --port 8080

The body is sent unchanged (model, messages, the 48 tool schemas, `stream`, `max_completion_tokens`), so the
server renders and tokenizes the same 18442-token prompt the client did.  The script prints the HTTP status,
whether the stream carried an error, the finish reason and the usage.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", nargs="?", default="request/request.json")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    with open(a.capture, encoding="utf-8") as f:
        body = json.load(f)
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
                    text.append((ch.get("delta") or {}).get("content") or "")
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                if ev.get("usage"):
                    usage = ev["usage"]
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}", flush=True)
        return 1
    out = "".join(text)
    print(f"status={status} {time.time() - t0:.1f}s finish={finish} usage={usage} chars={len(out)}", flush=True)
    if error:
        print(f"ERROR: {json.dumps(error, ensure_ascii=False)[:400]}", flush=True)
    print("text: " + out.replace("\n", " ")[:300], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
