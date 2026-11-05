#include "logs.h"

#include <sys/stat.h>

#include <fstream>

namespace ext {

std::string nccl_debug_file_path(const char *pattern, const std::string &host, int pid) {
  if (pattern == nullptr || *pattern == '\0') return "";
  // NCCL uses the host name up to the first dot.
  std::string short_host = host.substr(0, host.find('.'));
  std::string out;
  for (const char *p = pattern; *p; ++p) {
    if (*p != '%') { out.push_back(*p); continue; }
    char next = p[1];
    if (next == '\0') { out.push_back('%'); break; }
    ++p;
    if (next == '%') out.push_back('%');
    else if (next == 'h') out += short_host;
    else if (next == 'p') out += std::to_string(pid);
    else { out.push_back('%'); out.push_back(next); }
  }
  return out;
}

std::string artifact_name_part(const std::string &s) {
  std::string out;
  for (char c : s) {
    bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
    out.push_back(safe ? c : '_');
  }
  return out.empty() ? "unknown" : out;
}

bool read_file_capped(const std::string &path, size_t max_bytes, std::string *body, size_t *size) {
  struct stat st;
  if (path.empty() || stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return false;
  if (size) *size = (size_t)st.st_size;
  if ((size_t)st.st_size > max_bytes) return false;
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  // Read one byte past the limit at most, so a file that grew after stat is caught without
  // buffering what it grew by.
  body->resize(max_bytes + 1);
  f.read(&(*body)[0], (std::streamsize)body->size());
  body->resize((size_t)f.gcount());
  if (body->size() > max_bytes) { if (size) *size = body->size(); body->clear(); return false; }
  return true;
}

}  // namespace ext
