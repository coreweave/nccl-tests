#include "config.h"

#include <stdlib.h>
#include <string.h>

#include <cstdio>

#include "util.h"

namespace ext {

static const char *kUrlFileDefault = "/etc/perf-telemetry/url";

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

Config load_config(char **envp) {
  Config c;
  const char *sw = getenv("NCCL_TESTS_REPORT");
  if (sw && strcmp(sw, "0") == 0) return c;  // explicit opt-out

  const char *url = getenv("NCCL_TESTS_REPORT_URL");
  if (url) {
    c.url = trim(url);  // set but empty = force-disable
  } else {
    const char *file = getenv("NCCL_TESTS_REPORT_URL_FILE");
    c.url = read_first_line(file && *file ? file : kUrlFileDefault);
  }
  if (c.url.empty()) return c;
  c.active = true;

  const char *v;
  if ((v = getenv("NCCL_TESTS_REPORT_RUN_ID")) && *v) c.run_id_override = trim(v);
  if ((v = getenv("NCCL_TESTS_REPORT_AUTH_HEADER")) && *v) c.auth_header = v;
  if ((v = getenv("NCCL_TESTS_REPORT_CERT")) && *v) c.cert = v;
  if ((v = getenv("NCCL_TESTS_REPORT_KEY")) && *v) c.key = v;
  if ((v = getenv("NCCL_TESTS_REPORT_CA")) && *v) c.ca = v;
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

}  // namespace ext
