#include "rows.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "util.h"

namespace ext {

Value Value::number(double d) {
  Value v;
  if (std::isfinite(d)) { v.kind = NUM; v.num = d; }
  return v;
}

Value Value::text(const std::string &s) {
  Value v;
  if (!s.empty()) { v.kind = STR; v.str = clip_utf8(s, kMaxStringBytes); }
  return v;
}

void ctx_put(Context &c, const std::string &key, const std::string &s) {
  Value v = Value::text(s);
  if (!v.empty()) c.emplace_back(key, v);
}
void ctx_put(Context &c, const std::string &key, double d) {
  Value v = Value::number(d);
  if (!v.empty()) c.emplace_back(key, v);
}
void ctx_put(Context &c, const std::string &key, long long i) { ctx_put(c, key, (double)i); }

static bool is_reserved(const std::string &k) {
  static const char *kReserved[] = {"name", "value", "value_string", "kind", "version",
                                    "run_id", "region", "zone", "cluster", "cluster_org"};
  for (const char *r : kReserved) if (k == r) return true;
  return false;
}

static std::string sanitize_key(const std::string &in) {
  std::string k;
  k.reserve(in.size());
  for (unsigned char ch : in) {
    if (ch >= 'A' && ch <= 'Z') ch = (unsigned char)(ch - 'A' + 'a');
    if ((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_') k.push_back((char)ch);
    else k.push_back('_');
  }
  if (k.size() > kMaxKeyLen) k.resize(kMaxKeyLen);
  return k;
}

static bool key_ok(const std::string &k) {
  if (k.empty() || !(k[0] >= 'a' && k[0] <= 'z')) return false;
  return !is_reserved(k);
}

Context clean_context(const Context &in) {
  // Non-tag entries first, so tag_* are the ones dropped when the row is too wide.
  Context ordered;
  ordered.reserve(in.size());
  for (const auto &kv : in) if (kv.first.compare(0, 4, "tag_") != 0) ordered.push_back(kv);
  for (const auto &kv : in) if (kv.first.compare(0, 4, "tag_") == 0) ordered.push_back(kv);

  Context out;
  for (const auto &kv : ordered) {
    if (out.size() >= kMaxContextEntries) break;
    std::string k = sanitize_key(kv.first);
    if (!key_ok(k) || kv.second.empty()) continue;
    bool dup = false;
    for (const auto &o : out) if (o.first == k) { dup = true; break; }
    if (dup) continue;
    out.emplace_back(k, kv.second);
  }
  return out;
}

bool make_row(Row &out, const std::string &kind, const std::string &version,
              const std::string &name, const Value &v, const Context &base, const Context &extra) {
  if (name.empty() || v.empty()) return false;
  out.name = name;
  out.value = v;
  out.kind = kind;
  out.version = version;
  Context c = base;
  c.insert(c.end(), extra.begin(), extra.end());
  out.context = clean_context(c);
  return true;
}

static void json_str(std::string &o, const std::string &s) {
  o.push_back('"');
  for (unsigned char ch : s) {
    switch (ch) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (ch < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", ch); o += b; }
        else o.push_back((char)ch);
    }
  }
  o.push_back('"');
}

static void json_num(std::string &o, double d) {
  char b[40];
  if (std::floor(d) == d && std::fabs(d) < 1e15) snprintf(b, sizeof(b), "%.0f", d);
  else snprintf(b, sizeof(b), "%.17g", d);
  o += b;
}

static void json_value(std::string &o, const Value &v) {
  if (v.kind == Value::NUM) json_num(o, v.num);
  else json_str(o, v.str);
}

std::string rows_to_json(const std::vector<Row> &rows) {
  std::string o;
  o.reserve(rows.size() * 400);
  o.push_back('[');
  bool first = true;
  for (const Row &r : rows) {
    if (!first) o.push_back(',');
    first = false;
    o += "{\"name\":"; json_str(o, r.name);
    o += ",\"value\":"; json_value(o, r.value);
    o += ",\"kind\":"; json_str(o, r.kind);
    o += ",\"version\":"; json_str(o, r.version);
    o += ",\"context\":{";
    bool f2 = true;
    for (const auto &kv : r.context) {
      if (!f2) o.push_back(',');
      f2 = false;
      json_str(o, kv.first); o.push_back(':'); json_value(o, kv.second);
    }
    o += "}}";
  }
  o.push_back(']');
  return o;
}

}  // namespace ext
