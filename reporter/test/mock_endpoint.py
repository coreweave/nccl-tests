#!/usr/bin/env python3
"""Mock columnar-results endpoint for the tests.

Usage: mock_endpoint.py <port> <out_prefix>
Accepts PUT and saves the body to <out_prefix>.body, the request path and auth header
to <out_prefix>.meta, and the result of validating the rows against the columnar
results API field rules for name, value, kind, version and context to
<out_prefix>.validation, which reads PASS, or FAIL followed by the errors.
"""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT, OUT = int(sys.argv[1]), sys.argv[2]
KIND = re.compile(r"^[a-z][a-z0-9_-]*$")
CTX_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
RESERVED = {"name", "value", "value_string", "kind", "version", "run_id",
            "region", "zone", "cluster", "cluster_org"}


def validate(rows):
    errs = []
    if not isinstance(rows, list) or not rows:
        return ["empty or non-array body"]
    for i, r in enumerate(rows):
        if set(r) - {"name", "value", "kind", "version", "context"}:
            errs.append((i, "unknown field"))
        if not r.get("name"):
            errs.append((i, "name"))
        if isinstance(r.get("value"), str) and len(r["value"].encode()) > 32768:
            errs.append((i, "value longer than 32 KiB"))
        v = r.get("value")
        if isinstance(v, bool) or not (isinstance(v, (int, float)) or (isinstance(v, str) and v)):
            errs.append((i, "value type %r" % (v,)))
        if not KIND.match(str(r.get("kind", ""))) or len(r["kind"]) > 64:
            errs.append((i, "kind"))
        if not r.get("version"):
            errs.append((i, "version"))
        ctx = r.get("context", {})
        if len(ctx) > 64:
            errs.append((i, "more than 64 context keys"))
        for k, cv in ctx.items():
            if not CTX_KEY.match(k) or len(k) > 64 or k in RESERVED:
                errs.append((i, "context key " + k))
            if cv is None or isinstance(cv, (dict, list, bool)):
                errs.append((i, "context value for " + k))
            elif isinstance(cv, str) and len(cv.encode()) > 32768:
                errs.append((i, "context value longer than 32 KiB for " + k))
    return errs


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_PUT(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        try:
            errs = validate(json.loads(body))
        except Exception as ex:  # noqa: BLE001
            errs = ["bad json: %s" % ex]
        open(OUT + ".paths", "a").write(self.path + "\n")
        open(OUT + ".body", "wb").write(body)
        open(OUT + ".meta", "w").write("%s\nauth=%s\n" % (self.path, self.headers.get("X-Test", "")))
        open(OUT + ".validation", "w").write("PASS" if not errs else "FAIL %s" % errs[:5])
        self.send_response(202 if not errs else 400)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
