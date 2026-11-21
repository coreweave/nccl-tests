#include "secrets.h"

#include <ctype.h>
#include <string.h>
#include <strings.h>

namespace ext {

const char kRedacted[] = "<redacted>";

namespace {

// A name containing one of these parts holds a credential or an identity, unless it also
// contains an exempt part.
const char *const kSensitiveNameParts[] = {
    "KEY", "TOKEN", "SECRET", "PASS", "PWD", "CRED", "AUTH", "PRIVATE", "SIGNING", "SIGNATURE",
    "HMAC", "SALT", "JWT", "BEARER", "COOKIE", "SESSION", "LICENSE", "LICENCE", "DSN", "URL",
    "DATABASE", "REDIS", "MONGODB", "CONNECTION_STRING", "USERNAME", "USER", "EMAIL",
};
const char *const kExemptNameParts[] = {"PKEY"};

// Values that start with one of these are credentials issued by a known service.
const char *const kSecretValuePrefixes[] = {
    "Bearer ", "Basic ", "ghp_", "gho_", "github_pat_", "glpat-", "xoxb-", "xoxp-", "xoxa-",
    "AKIA", "ASIA", "sk-", "sk_live_", "sk_test_", "AIza", "ya29.", "eyJ",
};

// Inside a log a token has no clear end, so each prefix needs enough token characters after it
// to rule out ordinary words. AWS key ids are exactly 16 upper-case letters or digits after the
// prefix; the rest need at least min_rest.
struct TokenPrefix {
  const char *prefix;
  size_t min_rest;
  bool aws_key_id;
};
const TokenPrefix kTokenPrefixes[] = {
    {"github_pat_", 20, false}, {"ghp_", 20, false}, {"gho_", 20, false}, {"ghu_", 20, false},
    {"ghs_", 20, false},        {"ghr_", 20, false}, {"glpat-", 16, false}, {"xoxb-", 10, false},
    {"xoxp-", 10, false},       {"xoxa-", 10, false}, {"xoxs-", 10, false}, {"sk_live_", 16, false},
    {"sk_test_", 16, false},    {"sk-", 20, false},  {"AIza", 30, false}, {"ya29.", 20, false},
    {"AKIA", 16, true},         {"ASIA", 16, true},  {"eyJ", 20, false},
};

bool is_word(char c) { return isalnum((unsigned char)c) || c == '_'; }
bool is_name_char(char c) { return is_word(c) || c == '-'; }
bool is_token_char(char c) {
  return isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.' || c == '+' || c == '/' || c == '=';
}
bool is_space(char c) { return c == ' ' || c == '\t'; }

bool starts_with_ci(const std::string &s, size_t i, const char *p) {
  size_t n = strlen(p);
  return s.size() - i >= n && strncasecmp(s.c_str() + i, p, n) == 0;
}

size_t line_end(const std::string &s, size_t i) {
  size_t e = s.find('\n', i);
  return e == std::string::npos ? s.size() : e;
}

// The length of a known-service token at i, or 0.
size_t token_at(const std::string &s, size_t i) {
  for (const TokenPrefix &t : kTokenPrefixes) {
    size_t n = strlen(t.prefix);
    if (s.compare(i, n, t.prefix) != 0) continue;
    size_t j = i + n;
    if (t.aws_key_id) {
      size_t k = j;
      while (k < s.size() && k - j < 16 && (isupper((unsigned char)s[k]) || isdigit((unsigned char)s[k]))) ++k;
      if (k - j == 16 && (k == s.size() || !is_word(s[k]))) return k - i;
      continue;
    }
    while (j < s.size() && is_token_char(s[j])) ++j;
    if (j - i - n < t.min_rest) continue;
    // A JWT is three base64url parts separated by dots.
    if (t.prefix[0] == 'e' && s.find('.', i) >= j) continue;
    return j - i;
  }
  return 0;
}

// The length of a run of token characters at i.
size_t token_run(const std::string &s, size_t i) {
  size_t j = i;
  while (j < s.size() && is_token_char(s[j])) ++j;
  return j - i;
}

}  // namespace

bool is_sensitive_name(const std::string &name) {
  std::string upper = name;
  for (char &ch : upper) ch = (char)toupper((unsigned char)ch);
  bool sensitive = false;
  for (const char *p : kSensitiveNameParts) if (upper.find(p) != std::string::npos) { sensitive = true; break; }
  if (!sensitive) return false;
  for (const char *p : kExemptNameParts) if (upper.find(p) != std::string::npos) return false;
  return true;
}

bool has_secret_value_prefix(const std::string &value) {
  for (const char *p : kSecretValuePrefixes) {
    size_t n = strlen(p);
    if (value.size() >= n && value.compare(0, n, p) == 0) return true;
  }
  return false;
}

std::string scrub_log(const std::string &in, size_t *redactions) {
  std::string out;
  out.reserve(in.size());
  size_t count = 0;
  size_t i = 0;
  const size_t n = in.size();
  while (i < n) {
    char c = in[i];
    bool boundary = i == 0 || !is_name_char(in[i - 1]);

    // A PEM private key block, through its END line. Without an END line, the rest of the log.
    if (c == '-' && in.compare(i, 11, "-----BEGIN ") == 0) {
      size_t eol = line_end(in, i);
      if (in.substr(i, eol - i).find("PRIVATE KEY-----") != std::string::npos) {
        size_t end = in.find("-----END ", eol);
        size_t stop = n;
        if (end != std::string::npos) {
          size_t close = in.find("-----", end + 9);
          stop = close == std::string::npos ? line_end(in, end) : close + 5;
        }
        out += "<redacted private key>";
        ++count;
        i = stop;
        continue;
      }
    }

    // An Authorization header: everything after the colon on that line.
    if (boundary && (c == 'A' || c == 'a') && starts_with_ci(in, i, "authorization:")) {
      size_t j = i + 14;
      out.append(in, i, j - i);
      while (j < n && is_space(in[j])) out.push_back(in[j++]);
      size_t eol = line_end(in, j);
      if (eol > j) { out += kRedacted; ++count; }
      i = eol;
      continue;
    }

    // A Bearer token anywhere.
    if (boundary && (c == 'B' || c == 'b') && starts_with_ci(in, i, "bearer ")) {
      size_t j = i + 7;
      while (j < n && is_space(in[j])) ++j;
      size_t len = token_run(in, j);
      if (len >= 8) {
        out.append(in, i, j - i);
        out += kRedacted;
        ++count;
        i = j + len;
        continue;
      }
    }

    // A token that starts like a known service's.
    if (boundary) {
      size_t len = token_at(in, i);
      if (len) {
        out += kRedacted;
        ++count;
        i += len;
        continue;
      }
    }

    // The password in scheme://user:password@host.
    if (c == ':' && in.compare(i, 3, "://") == 0) {
      size_t j = i + 3, colon = std::string::npos;
      while (j < n && in[j] != '@' && in[j] != '/' && !isspace((unsigned char)in[j])) {
        if (in[j] == ':' && colon == std::string::npos) colon = j;
        ++j;
      }
      if (j < n && in[j] == '@' && colon != std::string::npos && j > colon + 1) {
        out.append(in, i, colon + 1 - i);
        out += kRedacted;
        ++count;
        i = j;
        continue;
      }
    }

    // NAME=value with a sensitive name, as in an environment dump or a --password=... flag.
    // A whole word is copied at once, so token checks never start inside one.
    if ((isalpha((unsigned char)c) || c == '_') && (i == 0 || !is_word(in[i - 1]))) {
      size_t j = i;
      while (j < n && is_name_char(in[j])) ++j;
      std::string name = in.substr(i, j - i);
      if (j < n && in[j] == '=' && is_sensitive_name(name)) {
        out.append(in, i, j + 1 - i);
        size_t k = j + 1;
        size_t stop;
        if (k < n && (in[k] == '"' || in[k] == '\'')) {
          size_t q = in.find(in[k], k + 1);
          size_t eol = line_end(in, k + 1);
          stop = (q == std::string::npos || q > eol) ? eol : q + 1;
        } else {
          stop = k;
          while (stop < n && !isspace((unsigned char)in[stop])) ++stop;
        }
        if (stop > k) { out += kRedacted; ++count; }
        i = stop;
        continue;
      }
      out += name;
      i = j;
      continue;
    }

    out.push_back(c);
    ++i;
  }
  if (redactions) *redactions = count;
  return out;
}

}  // namespace ext
