#include "config.h"

#include <netdb.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

#include <cstdio>
#include <sstream>

#include "util.h"

// Build-time defaults. The image build sets them. The library itself ships with none, so a
// build without them never reports unless the environment configures an endpoint.
#ifndef NCCL_TESTS_EXT_DEFAULT_HOSTS
#define NCCL_TESTS_EXT_DEFAULT_HOSTS ""
#endif
#ifndef NCCL_TESTS_EXT_SKIP_NAMESPACES
#define NCCL_TESTS_EXT_SKIP_NAMESPACES ""
#endif

namespace ext {

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

static bool host_resolves(const std::string &host) {
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo *res = nullptr;
  int rc = getaddrinfo(host.c_str(), nullptr, &hints, &res);
  if (res) freeaddrinfo(res);
  return rc == 0;
}

Config load_config(char **envp) {
  Config c;
  const char *sw = getenv("NCCL_TESTS_REPORT");
  if (sw && strcmp(sw, "0") == 0) { c.reason = "NCCL_TESTS_REPORT=0"; return c; }
  c.forced = sw && strcmp(sw, "1") == 0;

  const char *url = getenv("NCCL_TESTS_REPORT_URL");
  if (url) {
    c.url = trim(url);  // set but empty = force-disable
  } else {
    const char *file = getenv("NCCL_TESTS_REPORT_URL_FILE");
    c.url = read_first_line(file && *file ? file : kUrlFileDefault);
  }
  if (!c.url.empty()) {
    c.active = true;
    c.reason = "configured endpoint";
  } else {
    const char *v;
    c.candidate_hosts = split_csv((v = getenv("NCCL_TESTS_REPORT_HOSTS")) && *v ? v : NCCL_TESTS_EXT_DEFAULT_HOSTS);
    c.skip_namespaces = split_csv((v = getenv("NCCL_TESTS_REPORT_SKIP_NAMESPACES")) && *v ? v : NCCL_TESTS_EXT_SKIP_NAMESPACES);
    c.default_port = (int)env_long("NCCL_TESTS_REPORT_PORT", 4318);
    const char *rc = getenv("NCCL_TESTS_REPORT_RESOLV_CONF");
    std::vector<std::string> search = resolv_search_domains(rc && *rc ? rc : kResolvConfDefault);
    for (const std::string &d : search) if (d.find("cluster.local") != std::string::npos) c.in_cluster = true;
    c.pod_namespace = discover_namespace(search);
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
  if ((v = getenv("NCCL_VERSION")) && *v) c.nccl_version_env = v;
  c.connect_timeout_s = env_long("NCCL_TESTS_REPORT_CONNECT_TIMEOUT", 2);
  c.max_time_s = env_long("NCCL_TESTS_REPORT_MAX_TIME", 5);
  c.debug = (v = getenv("NCCL_TESTS_REPORT_DEBUG")) && strcmp(v, "1") == 0;

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
      } else if (key.compare(0, 5, "NCCL_") == 0 && key.compare(0, 17, "NCCL_TESTS_REPORT") != 0) {
        c.nccl_env.emplace_back(key, val);  // the reporter's own settings are not test config
      }
    }
  }
  return c;
}

bool resolve_default_endpoint(Config &c) {
  if (!c.pending) return c.active;
  for (const std::string &h : c.candidate_hosts) {
    if (host_resolves(h)) {
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
