#include "http.h"

#include <dlfcn.h>

#include <cstdio>

namespace ext {

// libcurl ABI constants. They have been stable since libcurl 7.x. The headers are avoided on
// purpose so the build needs no curl-dev package.
namespace curl {
typedef void CURL;
typedef int CURLcode;
struct slist;
enum { CURLE_OK = 0 };
enum : long {
  CURL_GLOBAL_DEFAULT = 3,
  CURLOPT_WRITEDATA = 10001,
  CURLOPT_URL = 10002,
  CURLOPT_TIMEOUT = 13,
  CURLOPT_POSTFIELDS = 10015,
  CURLOPT_WRITEFUNCTION = 20011,
  CURLOPT_HTTPHEADER = 10023,
  CURLOPT_CUSTOMREQUEST = 10036,
  CURLOPT_FAILONERROR = 45,
  CURLOPT_POSTFIELDSIZE = 60,
  CURLOPT_CONNECTTIMEOUT = 78,
  CURLOPT_NOSIGNAL = 99,
  CURLINFO_RESPONSE_CODE = 0x200002,
};
typedef CURLcode (*global_init_t)(long);
typedef CURL *(*easy_init_t)(void);
typedef CURLcode (*easy_setopt_t)(CURL *, int, ...);
typedef CURLcode (*easy_perform_t)(CURL *);
typedef void (*easy_cleanup_t)(CURL *);
typedef CURLcode (*easy_getinfo_t)(CURL *, int, ...);
typedef const char *(*easy_strerror_t)(CURLcode);
typedef slist *(*slist_append_t)(slist *, const char *);
typedef void (*slist_free_all_t)(slist *);

struct Api {
  void *handle = nullptr;
  global_init_t global_init = nullptr;
  easy_init_t easy_init = nullptr;
  easy_setopt_t easy_setopt = nullptr;
  easy_perform_t easy_perform = nullptr;
  easy_cleanup_t easy_cleanup = nullptr;
  easy_getinfo_t easy_getinfo = nullptr;
  easy_strerror_t easy_strerror = nullptr;
  slist_append_t slist_append = nullptr;
  slist_free_all_t slist_free_all = nullptr;
  bool ok() const {
    return handle && global_init && easy_init && easy_setopt && easy_perform && easy_cleanup &&
           easy_getinfo && easy_strerror && slist_append && slist_free_all;
  }
};

static Api load() {
  static const char *names[] = {"libcurl.so.4", "libcurl-gnutls.so.4", "libcurl-nss.so.4", "libcurl.so"};
  Api a;
  for (const char *n : names) {
    a.handle = dlopen(n, RTLD_NOW | RTLD_LOCAL);
    if (a.handle) break;
  }
  if (!a.handle) return a;
#define SYM(field, name) a.field = reinterpret_cast<decltype(a.field)>(dlsym(a.handle, name))
  SYM(global_init, "curl_global_init");
  SYM(easy_init, "curl_easy_init");
  SYM(easy_setopt, "curl_easy_setopt");
  SYM(easy_perform, "curl_easy_perform");
  SYM(easy_cleanup, "curl_easy_cleanup");
  SYM(easy_getinfo, "curl_easy_getinfo");
  SYM(easy_strerror, "curl_easy_strerror");
  SYM(slist_append, "curl_slist_append");
  SYM(slist_free_all, "curl_slist_free_all");
#undef SYM
  return a;
}

static size_t discard(char *, size_t s, size_t n, void *) { return s * n; }
}  // namespace curl

bool http_put_json(const std::string &url, const std::string &body, const HttpOptions &opt, std::string *err) {
  using namespace curl;
  static Api api = load();  // one dlopen per process
  if (!api.ok()) { if (err) *err = "libcurl not available"; return false; }

  static bool global_done = false;
  if (!global_done) { api.global_init(CURL_GLOBAL_DEFAULT); global_done = true; }

  CURL *h = api.easy_init();
  if (!h) { if (err) *err = "curl_easy_init failed"; return false; }

  slist *headers = api.slist_append(nullptr, "Content-Type: application/json");
  headers = api.slist_append(headers, "Expect:");  // no 100-continue round trip
  if (!opt.auth_header.empty()) headers = api.slist_append(headers, opt.auth_header.c_str());

  api.easy_setopt(h, CURLOPT_URL, url.c_str());
  api.easy_setopt(h, CURLOPT_CUSTOMREQUEST, "PUT");
  api.easy_setopt(h, CURLOPT_POSTFIELDS, body.data());
  api.easy_setopt(h, CURLOPT_POSTFIELDSIZE, (long)body.size());
  api.easy_setopt(h, CURLOPT_HTTPHEADER, headers);
  api.easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
  api.easy_setopt(h, CURLOPT_FAILONERROR, 1L);
  api.easy_setopt(h, CURLOPT_CONNECTTIMEOUT, opt.connect_timeout_s);
  api.easy_setopt(h, CURLOPT_TIMEOUT, opt.max_time_s);
  api.easy_setopt(h, CURLOPT_WRITEFUNCTION, &discard);
  api.easy_setopt(h, CURLOPT_WRITEDATA, nullptr);

  CURLcode rc = api.easy_perform(h);
  long status = 0;
  api.easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
  bool ok = (rc == CURLE_OK) && status >= 200 && status < 300;
  if (!ok && err) {
    char b[160];
    snprintf(b, sizeof(b), "%s (http %ld)", api.easy_strerror(rc), status);
    *err = b;
  }
  api.slist_free_all(headers);
  api.easy_cleanup(h);
  return ok;
}

}  // namespace ext
