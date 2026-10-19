"""A logging reverse proxy: everything DSH sends to :8080 is written to disk, then forwarded to the real Strata
server on :8081 (and the answer streamed straight back).  No change to Strata or to DSH - the client still talks to
the address it was configured with."""
import http.client
import http.server
import itertools
import json
import os
import sys
import time

LISTEN = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
UPSTREAM = ("127.0.0.1", int(sys.argv[2]) if len(sys.argv) > 2 else 8081)
DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
count = itertools.count(1)


def summary(body):
    try:
        req = json.loads(body)
    except ValueError:
        return ""
    if not isinstance(req, dict):
        return ""
    msgs = req.get("messages")
    parts = [f"messages={len(msgs) if isinstance(msgs, list) else '?'}",
             f"tools={len(req.get('tools') or [])}",
             f"stream={bool(req.get('stream'))}",
             f"max_tokens={req.get('max_tokens') or req.get('max_completion_tokens')}",
             f"model={req.get('model')!r}"]
    roles = [m.get("role") for m in (msgs or []) if isinstance(m, dict)]
    parts.append("roles=" + ",".join(str(r) for r in roles[:12]) + ("..." if len(roles) > 12 else ""))
    return " ".join(parts)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _relay(self, method):
        n = next(count)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if body:
            os.makedirs(DUMP, exist_ok=True)
            path = os.path.join(DUMP, f"{time.strftime('%Y%m%d-%H%M%S')}-{n:03d}-{method}-{self.path.strip('/').replace('/', '_')}.json")
            with open(path, "wb") as f:
                f.write(body)
            print(f"[proxy] {method} {self.path} {len(body)} B -> {path}\n[proxy]   {summary(body)}", flush=True)
        else:
            print(f"[proxy] {method} {self.path} (no body)", flush=True)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection")}
        conn = http.client.HTTPConnection(*UPSTREAM, timeout=3600)
        try:
            conn.request(method, self.path, body=body or None, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in ("transfer-encoding", "connection", "content-length"):
                    continue
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            status = resp.status
            total = 0
            while True:
                chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                total += len(chunk)
            print(f"[proxy] <- {status} {total} B", flush=True)
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"[proxy] client left: {e}", flush=True)
        finally:
            conn.close()

    def do_GET(self):
        self._relay("GET")

    def do_POST(self):
        self._relay("POST")

    def do_DELETE(self):
        self._relay("DELETE")


if __name__ == "__main__":
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", LISTEN), Handler)
    srv.daemon_threads = True
    print(f"[proxy] 127.0.0.1:{LISTEN} -> 127.0.0.1:{UPSTREAM[1]}, dumping to " + DUMP, flush=True)
    srv.serve_forever()
