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

// Moves a cut back, or forward, off the continuation bytes of a UTF-8 sequence.
static size_t back_to_boundary(const std::string &s, size_t cut) {
  while (cut > 0 && cut < s.size() && (((unsigned char)s[cut]) & 0xC0) == 0x80) --cut;
  return cut;
}
static size_t forward_to_boundary(const std::string &s, size_t start) {
  while (start < s.size() && (((unsigned char)s[start]) & 0xC0) == 0x80) ++start;
  return start;
}

bool read_head_tail(const std::string &path, size_t max_bytes, std::string *text, size_t *size, bool *truncated) {
  *truncated = false;
  struct stat st;
  if (path.empty() || stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return false;
  if (size) *size = (size_t)st.st_size;
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  // One byte past the limit shows whether the file, which may still grow, fits.
  std::string head(max_bytes + 1, '\0');
  f.read(&head[0], (std::streamsize)head.size());
  head.resize((size_t)f.gcount());
  if (head.size() <= max_bytes) { *text = head; return true; }
  *truncated = true;
  f.clear();
  f.seekg(0, std::ios::end);
  size_t total = (size_t)f.tellg();
  std::string marker_probe = "\n[... " + std::to_string(total) + " bytes omitted ...]\n";
  size_t keep = max_bytes > marker_probe.size() ? max_bytes - marker_probe.size() : 0;
  size_t head_len = back_to_boundary(head, keep / 2);
  size_t tail_len = keep - keep / 2;
  std::string tail(tail_len, '\0');
  f.seekg((std::streamoff)(total - tail_len), std::ios::beg);
  f.read(&tail[0], (std::streamsize)tail_len);
  tail.resize((size_t)f.gcount());
  tail = tail.substr(forward_to_boundary(tail, 0));
  std::string marker = "\n[... " + std::to_string(total - head_len - tail.size()) + " bytes omitted ...]\n";
  *text = head.substr(0, head_len) + marker + tail;
  return true;
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
