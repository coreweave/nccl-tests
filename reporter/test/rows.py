#!/usr/bin/env python3
"""Query helper for run_tests.sh: rows.py <body.json> <cmd> [args]

  count KIND [NAME]                number of rows
  value KIND NAME [k=v ...]        value of the first row whose context matches k=v
  ctx   KIND NAME KEY [k=v ...]    context[KEY] of that row
  names KIND                       sorted unique names
  maxstr                           longest string (bytes) across values and contexts
  maxkeys                          most context keys on one row
"""
import json
import sys

rows = json.load(open(sys.argv[1]))
cmd, a = sys.argv[2], sys.argv[3:]


def matches(r, kind, name, conds):
    if r["kind"] != kind or (name is not None and r["name"] != name):
        return False
    for c in conds:
        k, v = c.split("=", 1)
        if str(r["context"].get(k)) != v:
            return False
    return True


def first(kind, name, conds):
    for r in rows:
        if matches(r, kind, name, conds):
            return r
    sys.exit("no row: %s %s %s" % (kind, name, conds))


if cmd == "count":
    print(sum(1 for r in rows if matches(r, a[0], a[1] if len(a) > 1 else None, [])))
elif cmd == "value":
    print(first(a[0], a[1], a[2:])["value"])
elif cmd == "ctx":
    print(first(a[0], a[1], a[3:])["context"].get(a[2]))
elif cmd == "names":
    print(" ".join(sorted({r["name"] for r in rows if r["kind"] == a[0]})))
elif cmd == "maxstr":
    print(max([len(str(r["value"]).encode()) for r in rows] +
              [len(str(v).encode()) for r in rows for v in r["context"].values() if isinstance(v, str)]))
elif cmd == "maxkeys":
    print(max(len(r["context"]) for r in rows))
else:
    sys.exit("unknown command")
