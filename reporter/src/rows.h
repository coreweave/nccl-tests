// The row model of the columnar results API and its client-side limits.
// A row is {name, value, kind, version, context}. The API rejects the whole request on
// one bad row, so every rule is enforced here before anything is sent.
#pragma once
#include <string>
#include <utility>
#include <vector>

namespace ext {

struct Value {
  enum Kind { NONE, NUM, STR } kind = NONE;
  double num = 0;
  std::string str;
  static Value number(double d);
  static Value text(const std::string &s);
  bool empty() const { return kind == NONE; }
};

using Context = std::vector<std::pair<std::string, Value>>;

struct Row {
  std::string name;
  Value value;
  std::string kind;
  std::string version;
  Context context;
};

// Limits of the API.
constexpr size_t kMaxKeyLen = 64;
constexpr size_t kMaxContextEntries = 64;
constexpr size_t kMaxStringBytes = 32768;

// Add to a context, dropping empty strings and non-finite numbers.
void ctx_put(Context &c, const std::string &key, const std::string &s);
void ctx_put(Context &c, const std::string &key, double d);
void ctx_put(Context &c, const std::string &key, long long i);

// Enforce the API rules on a context. Keys become lowercase [a-z0-9_] starting with a
// letter, reserved names are dropped, at most 64 entries remain with tag_* dropped first,
// and strings are clipped.
Context clean_context(const Context &in);

// Build a row, or return false when the value is an empty string or not a finite number.
bool make_row(Row &out, const std::string &kind, const std::string &version,
              const std::string &name, const Value &v, const Context &base, const Context &extra);

// Compact JSON array of rows.
std::string rows_to_json(const std::vector<Row> &rows);

}  // namespace ext
