#include "config.h"

#include <netdb.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>

#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "util.h"

// Build-time defaults. The image build sets them. The library itself ships with none, so a
// build without them never reports unless the environment configures an endpoint.
#ifndef NCCL_TESTS_EXT_DEFAULT_HOSTS
#define NCCL_TESTS_EXT_DEFAULT_HOSTS ""
#endif
#ifndef NCCL_TESTS_EXT_SKIP_NAMESPACES
#define NCCL_TESTS_EXT_SKIP_NAMESPACES ""
#endif
#ifndef NCCL_TESTS_EXT_ARTIFACT_HOSTS
#define NCCL_TESTS_EXT_ARTIFACT_HOSTS ""
#endif

namespace ext {

// Environment variables are recorded by name prefix. These prefixes cover the libraries that
// decide how NCCL moves data. Nothing outside them is read.
static const char *const kRecordedPrefixes[] = {
    "NCCL_", "UCX_", "UCC_", "HCOLL_", "SHARP_",
    "OMPI_", "OPAL_", "PRTE_", "PMIX_", "PMI_", "MPICH_", "I_MPI_",
    "CUDA_", "GDRCOPY_", "MLX5_", "RDMAV_", "IBV_",
};

// A name containing one of these parts holds a credential or an identity and is dropped,
// unless it also contains an exempt part. InfiniBand partition keys are settings, not secrets.
static const char *const kSensitiveNameParts[] = {
    "KEY", "TOKEN", "SECRET", "PASS", "PWD", "CRED", "AUTH", "PRIVATE", "SIGNING", "SIGNATURE",
    "HMAC", "SALT", "JWT", "BEARER", "COOKIE", "SESSION", "LICENSE", "LICENCE", "DSN", "URL",
    "DATABASE", "REDIS", "MONGODB", "CONNECTION_STRING", "USERNAME", "USER", "EMAIL",
};
static const char *const kExemptNameParts[] = {"PKEY"};

// A value that starts with one of these is a credential issued by a known service. The
// variable is kept and the value replaced, so the setting is known to exist without its content.
static const char *const kSecretValuePrefixes[] = {
    "Bearer ", "Basic ", "ghp_", "gho_", "github_pat_", "glpat-", "xoxb-", "xoxp-", "xoxa-",
    "AKIA", "ASIA", "sk-", "sk_live_", "sk_test_", "AIza", "ya29.", "eyJ",
};
static const char kRedactedValue[] = "<redacted>";

// Name comparisons ignore case.
template <size_t N>
static bool name_starts_with(const std::string &name, const char *const (&prefixes)[N]) {
  for (const char *p : prefixes) {
    size_t n = strlen(p);
    if (name.size() >= n && strncasecmp(name.c_str(), p, n) == 0) return true;
  }
  return false;
}

template <size_t N>
static bool name_contains(const std::string &name, const char *const (&parts)[N]) {
  std::string upper = name;
  for (char &ch : upper) ch = (char)toupper((unsigned char)ch);
  for (const char *p : parts) if (upper.find(p) != std::string::npos) return true;
  return false;
}

// True when a set variable belongs in the config rows. The reporter's own settings never do.
static bool is_recorded(const std::string &key) {
  if (strncasecmp(key.c_str(), "NCCL_TESTS_REPORT", 17) == 0) return false;
  if (!name_starts_with(key, kRecordedPrefixes)) return false;
  if (name_contains(key, kSensitiveNameParts) && !name_contains(key, kExemptNameParts)) return false;
  return true;
}

static std::string recorded_value(const std::string &val) {
  for (const char *p : kSecretValuePrefixes) {
    size_t n = strlen(p);
    if (val.size() >= n && val.compare(0, n, p) == 0) return kRedactedValue;
  }
  return val;
}

static const char *kUrlFileDefault = "/etc/perf-telemetry/url";
static const char *kNamespaceFileDefault = "/var/run/secrets/kubernetes.io/serviceaccount/namespace";
static const char *kResolvConfDefault = "/etc/resolv.conf";

static std::string read_first_line(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return "";
  char buf[4096];
  std::string line;
  if (fgets(buf, sizeof(buf), f)) line = buf;
  fclose(f);
  size_t nl = line.find_first_of("\r\n");
  if (nl != std::string::npos) line.erase(nl);
  return trim(line);
}

static long env_long(const char *name, long dflt) {
  const char *v = getenv(name);
  if (!v || !*v) return dflt;
  char *end = nullptr;
  long x = strtol(v, &end, 10);
  if (end == v || x < 0) return dflt;
  return x;
}

// Like env_long but zero is rejected too, for timeouts where zero means no limit to libcurl.
static long env_positive_long(const char *name, long dflt) {
  long x = env_long(name, dflt);
  return x > 0 ? x : dflt;
}

static std::vector<std::string> split_csv(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim(item);
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

// The search domains of resolv.conf, in order. Empty when the file is missing.
static std::vector<std::string> resolv_search_domains(const char *path) {
  std::vector<std::string> out;
  FILE *f = fopen(path, "r");
  if (!f) return out;
  char buf[4096];
  while (fgets(buf, sizeof(buf), f)) {
    std::string line = trim(buf);
    if (line.compare(0, 7, "search ") != 0) continue;
    std::stringstream ss(line.substr(7));
    std::string d;
    while (ss >> d) out.push_back(d);
    break;
  }
  fclose(f);
  return out;
}

// The pod namespace. The service account file is authoritative when it is mounted. Otherwise
// the first search domain is <namespace>.svc.cluster.local in every Kubernetes pod.
static std::string discover_namespace(const std::vector<std::string> &search) {
  const char *v = getenv("NCCL_TESTS_REPORT_NAMESPACE");
  if (v && *v) return trim(v);
  const char *file = getenv("NCCL_TESTS_REPORT_NAMESPACE_FILE");
  std::string ns = read_first_line(file && *file ? file : kNamespaceFileDefault);
  if (!ns.empty()) return ns;
  static const std::string suffix = ".svc.cluster.local";
  for (const std::string &d : search) {
    if (d.size() > suffix.size() && d.compare(d.size() - suffix.size(), suffix.size(), suffix) == 0) {
      std::string head = d.substr(0, d.size() - suffix.size());
      if (head.find('.') == std::string::npos) return head;
    }
  }
  return "";
}

// Resolves host on a detached thread and waits until deadline. A lookup still running at the
// deadline counts as not resolved and is left to finish on its own, so a slow resolver never
// holds up the end of the test.
struct Lookup {
  std::mutex m;
  std::condition_variable cv;
  bool done = false, ok = false;
};

static bool host_resolves(const std::string &host, std::chrono::steady_clock::time_point deadline) {
  if (std::chrono::steady_clock::now() >= deadline) return false;
  auto l = std::make_shared<Lookup>();
  try {
    std::thread([l, host]() {
      struct addrinfo hints;
      memset(&hints, 0, sizeof(hints));
      hints.ai_family = AF_UNSPEC;
      hints.ai_socktype = SOCK_STREAM;
      struct addrinfo *res = nullptr;
      int rc = getaddrinfo(host.c_str(), nullptr, &hints, &res);
      if (res) freeaddrinfo(res);
      std::lock_guard<std::mutex> g(l->m);
      l->ok = rc == 0;
      l->done = true;
      l->cv.notify_all();
    }).detach();
  } catch (...) {
    return false;
  }
  std::unique_lock<std::mutex> g(l->m);
  l->cv.wait_until(g, deadline, [&] { return l->done; });
  return l->done && l->ok;
}

Config load_config(char **envp) {
  Config c;
  const char *sw = getenv("NCCL_TESTS_REPORT");
  if (sw && strcmp(sw, "0") == 0) { c.reason = "NCCL_TESTS_REPORT=0"; return c; }
  c.forced = sw && strcmp(sw, "1") == 0;

  const char *url = getenv("NCCL_TESTS_REPORT_URL");
  if (url) {
    c.url = trim(url);
    if (c.url.empty()) { c.reason = "NCCL_TESTS_REPORT_URL is set but empty"; return c; }
  } else {
    const char *file = getenv("NCCL_TESTS_REPORT_URL_FILE");
    c.url = read_first_line(file && *file ? file : kUrlFileDefault);
  }
  // The pod namespace is discovered on every path, because it is recorded with the rows and
  // not only consulted for the skip list.
  {
    const char *rc = getenv("NCCL_TESTS_REPORT_RESOLV_CONF");
    std::vector<std::string> search = resolv_search_domains(rc && *rc ? rc : kResolvConfDefault);
    for (const std::string &d : search) if (d.find("cluster.local") != std::string::npos) c.in_cluster = true;
    c.pod_namespace = discover_namespace(search);
  }
  if (!c.url.empty()) {
    c.active = true;
    c.reason = "configured endpoint";
  } else {
    const char *v;
    c.candidate_hosts = split_csv((v = getenv("NCCL_TESTS_REPORT_HOSTS")) && *v ? v : NCCL_TESTS_EXT_DEFAULT_HOSTS);
    c.skip_namespaces = split_csv((v = getenv("NCCL_TESTS_REPORT_SKIP_NAMESPACES")) && *v ? v : NCCL_TESTS_EXT_SKIP_NAMESPACES);
    c.default_port = (int)env_long("NCCL_TESTS_REPORT_PORT", 4318);
    bool skipped = false;
    for (const std::string &n : c.skip_namespaces) if (!c.pod_namespace.empty() && n == c.pod_namespace) skipped = true;
    if (skipped && !c.forced) { c.reason = "namespace " + c.pod_namespace + " is on the skip list"; return c; }
    if (c.candidate_hosts.empty()) { c.reason = "no endpoint configured and no candidate hosts"; return c; }
    if (!c.in_cluster && !c.forced) { c.reason = "not in a cluster, no cluster.local search domain"; return c; }
    c.active = true;
    c.pending = true;
    c.reason = c.forced ? "forced on, endpoint resolved at finalize" : "in a cluster, endpoint resolved at finalize";
  }

  const char *v;
  if ((v = getenv("NCCL_TESTS_REPORT_RUN_ID")) && *v) c.run_id_override = trim(v);
  if ((v = getenv("NCCL_TESTS_REPORT_AUTH_HEADER")) && *v) c.auth_header = v;
  if ((v = getenv("NCCL_TESTS_REPORT_IMAGE")) && *v) c.image = v;
  if ((v = getenv("SLURM_JOB_ID")) && *v) c.slurm_job_id = v;
  c.connect_timeout_s = env_positive_long("NCCL_TESTS_REPORT_CONNECT_TIMEOUT", 2);
  c.max_time_s = env_positive_long("NCCL_TESTS_REPORT_MAX_TIME", 5);
  c.debug = (v = getenv("NCCL_TESTS_REPORT_DEBUG")) && strcmp(v, "1") == 0;

  // Log upload is a separate opt-in on top of an active reporter.
  c.logs = (v = getenv("NCCL_TESTS_REPORT_LOGS")) && strcmp(v, "1") == 0;
  if (c.logs) {
    c.log_max_time_s = env_positive_long("NCCL_TESTS_REPORT_LOG_MAX_TIME", 30);
    long max_mib = env_positive_long("NCCL_TESTS_REPORT_LOG_MAX_MIB", 64);
    c.log_max_bytes = (size_t)(max_mib < 256 ? max_mib : 256) << 20;
    if ((v = getenv("NCCL_TESTS_REPORT_ARTIFACT_URL")) && *v) {
      c.artifact_url = trim(v);
    } else {
      c.artifact_hosts = split_csv((v = getenv("NCCL_TESTS_REPORT_ARTIFACT_HOSTS")) && *v ? v : NCCL_TESTS_EXT_ARTIFACT_HOSTS);
      c.default_port = (int)env_long("NCCL_TESTS_REPORT_PORT", 4318);
      const char *rc = getenv("NCCL_TESTS_REPORT_RESOLV_CONF");
      bool in_cluster = c.in_cluster;
      for (const std::string &d : resolv_search_domains(rc && *rc ? rc : kResolvConfDefault))
        if (d.find("cluster.local") != std::string::npos) in_cluster = true;
      c.artifact_pending = !c.artifact_hosts.empty() && (in_cluster || c.forced);
    }
  }

  static const char kTag[] = "NCCL_TESTS_REPORT_TAG_";
  const size_t tag_len = sizeof(kTag) - 1;
  if (envp) {
    for (char **e = envp; *e; ++e) {
      const char *eq = strchr(*e, '=');
      if (!eq) continue;
      std::string key(*e, (size_t)(eq - *e));
      std::string val(eq + 1);
      if (key.compare(0, tag_len, kTag) == 0 && key.size() > tag_len) {
        c.tags.emplace_back(key.substr(tag_len), val);
      } else if (is_recorded(key)) {
        c.recorded_env.emplace_back(key, recorded_value(val));
      }
    }
  }
  return c;
}

bool resolve_artifact_endpoint(Config &c) {
  if (!c.artifact_url.empty()) return true;
  if (!c.artifact_pending) return false;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(c.connect_timeout_s);
  for (const std::string &h : c.artifact_hosts) {
    if (host_resolves(h, deadline)) {
      c.artifact_url = "http://" + h + ":" + std::to_string(c.default_port) + c.artifact_path;
      c.artifact_pending = false;
      return true;
    }
  }
  return false;
}

bool resolve_default_endpoint(Config &c) {
  if (!c.pending) return c.active;
  // All candidates share one budget, the connect timeout.
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(c.connect_timeout_s);
  for (const std::string &h : c.candidate_hosts) {
    if (host_resolves(h, deadline)) {
      c.url = "http://" + h + ":" + std::to_string(c.default_port) + c.default_path;
      c.pending = false;
      c.reason = "resolved " + h;
      return true;
    }
  }
  if (c.forced && !c.candidate_hosts.empty()) {
    c.url = "http://" + c.candidate_hosts[0] + ":" + std::to_string(c.default_port) + c.default_path;
    c.pending = false;
    c.reason = "forced on, no candidate resolved, using " + c.candidate_hosts[0];
    return true;
  }
  c.pending = false;
  c.active = false;
  c.reason = "no candidate host resolved";
  return false;
}

}  // namespace ext
