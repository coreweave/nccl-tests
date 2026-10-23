// One HTTP PUT of a JSON body through libcurl, which is loaded with dlopen at call
// time. A missing libcurl means "not delivered", never a build or load failure of
// the test binary. One attempt, short timeouts, no signals.
#pragma once
#include <string>

namespace ext {

struct HttpOptions {
  long connect_timeout_s = 2;
  long max_time_s = 5;
  std::string auth_header;   // full header line, e.g. "Authorization: Bearer x"
};

// Returns true when the server answered 2xx. err gets a short reason otherwise.
bool http_put_json(const std::string &url, const std::string &body, const HttpOptions &opt, std::string *err);

}  // namespace ext
