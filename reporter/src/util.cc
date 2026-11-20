#include "util.h"

#include <sys/random.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>

namespace ext {

timespec now_realtime() {
  timespec t{};
  clock_gettime(CLOCK_REALTIME, &t);
  return t;
}

double seconds_between(const timespec &a, const timespec &b) {
  return (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) / 1e9;
}

std::string iso_utc_ms(const timespec &t) {
  struct tm tm {};
  time_t secs = t.tv_sec;
  if (gmtime_r(&secs, &tm) == nullptr) return "";
  char buf[40];
  int n = snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ",
                   tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                   tm.tm_hour, tm.tm_min, tm.tm_sec, t.tv_nsec / 1000000L);
  return n > 0 ? std::string(buf, (size_t)n) : "";
}

static void fill_random(unsigned char *p, size_t n) {
  size_t got = 0;
  while (got < n) {
    ssize_t r = getrandom(p + got, n - got, 0);
    if (r <= 0) break;
    got += (size_t)r;
  }
  if (got < n) {
    // Fall back to the clock. Uniqueness within a process is still fine.
    timespec t = now_realtime();
    unsigned long long x = (unsigned long long)t.tv_sec * 1000000007ULL ^ (unsigned long long)t.tv_nsec ^ (unsigned long long)getpid();
    for (; got < n; ++got) {
      x ^= x << 13; x ^= x >> 7; x ^= x << 17;
      p[got] = (unsigned char)(x & 0xff);
    }
  }
}

std::string uuid_v7() {
  unsigned char b[16];
  fill_random(b, sizeof(b));
  timespec t = now_realtime();
  unsigned long long ms = (unsigned long long)t.tv_sec * 1000ULL + (unsigned long long)(t.tv_nsec / 1000000L);
  for (int i = 0; i < 6; ++i) b[i] = (unsigned char)((ms >> (8 * (5 - i))) & 0xff);
  b[6] = (unsigned char)((b[6] & 0x0f) | 0x70);  // version 7
  b[8] = (unsigned char)((b[8] & 0x3f) | 0x80);  // variant 10xx
  char out[37];
  snprintf(out, sizeof(out),
           "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
  return out;
}

std::string hostname() {
  char buf[256];
  if (gethostname(buf, sizeof(buf)) != 0) return "";
  buf[sizeof(buf) - 1] = '\0';
  return buf;
}

std::string humanize_size(unsigned long long n) {
  struct { const char *u; unsigned long long div; } units[] = {{"G", 1ULL << 30}, {"M", 1ULL << 20}, {"K", 1ULL << 10}};
  char buf[48];
  for (auto &x : units) {
    if (n >= x.div) {
      double q = (double)n / (double)x.div;
      if (std::floor(q) == q) snprintf(buf, sizeof(buf), "%.0f%s", q, x.u);
      else snprintf(buf, sizeof(buf), "%.1f%s", q, x.u);
      return buf;
    }
  }
  snprintf(buf, sizeof(buf), "%llu", n);
  return buf;
}

std::string dotted_nccl_version(int code) {
  if (code < 20000) return "";
  char buf[32];
  snprintf(buf, sizeof(buf), "%d.%d.%d", code / 10000, code / 100 % 100, code % 100);
  return buf;
}

std::string cuda_major_minor(int cudart) {
  if (cudart < 1000) return "";
  char buf[32];
  snprintf(buf, sizeof(buf), "%d.%d", cudart / 1000, (cudart % 1000) / 10);
  return buf;
}

std::string trim(const std::string &s) {
  size_t a = 0, b = s.size();
  while (a < b && isspace((unsigned char)s[a])) ++a;
  while (b > a && isspace((unsigned char)s[b - 1])) --b;
  return s.substr(a, b - a);
}

std::string clip_utf8(const std::string &s, size_t max_bytes) {
  if (s.size() <= max_bytes) return s;
  size_t n = max_bytes;
  while (n > 0 && ((unsigned char)s[n] & 0xC0) == 0x80) --n;  // do not split a code point
  return s.substr(0, n);
}

}  // namespace ext
