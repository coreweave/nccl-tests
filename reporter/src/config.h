// Runtime configuration of the reporter. Everything comes from the environment or a
// mounted file, plus two build-time defaults, the candidate collector hosts and the namespaces that
// never report. With neither a configured url nor a resolvable candidate host the reporter is inactive.
#pragma once
#include <string>
#include <utility>
#include <vector>

namespace ext {

struct Config {
  bool active = false;          // when false, every hook is a no-op with no files and no network
  bool pending = false;         // active, but the endpoint is decided at finalize by resolving candidate hosts
  bool forced = false;          // NCCL_TESTS_REPORT=1
  std::string reason;           // why the reporter is active or not, for the debug log
  std::vector<std::string> candidate_hosts;  // tried in order when pending, from the build default or NCCL_TESTS_REPORT_HOSTS
  int default_port = 4318;      // plain listener of the collector, NCCL_TESTS_REPORT_PORT
  std::string default_path = "/v2/runs/{run_id}/results";
  std::vector<std::string> skip_namespaces;  // pod namespaces that never report, from the build default or NCCL_TESTS_REPORT_SKIP_NAMESPACES
  std::string pod_namespace;    // from the service account file, else the first resolv.conf search domain
  bool in_cluster = false;      // resolv.conf has a cluster.local search domain
  std::string url;              // may contain {run_id}
  std::string run_id_override;  // NCCL_TESTS_REPORT_RUN_ID
  std::string auth_header;      // NCCL_TESTS_REPORT_AUTH_HEADER, sent verbatim as one header line
  std::string image;            // NCCL_TESTS_REPORT_IMAGE
  long connect_timeout_s = 2;   // NCCL_TESTS_REPORT_CONNECT_TIMEOUT
  long max_time_s = 5;          // NCCL_TESTS_REPORT_MAX_TIME
  bool debug = false;           // NCCL_TESTS_REPORT_DEBUG=1 logs to stderr
  std::vector<std::pair<std::string, std::string>> tags;  // NCCL_TESTS_REPORT_TAG_<NAME> as name and value
  std::vector<std::pair<std::string, std::string>> recorded_env;  // set variables under the recorded prefixes, credentials removed
  std::string slurm_job_id;     // SLURM_JOB_ID
  bool logs = false;            // NCCL_TESTS_REPORT_LOGS=1 uploads log files as run artifacts
  std::string artifact_url;     // NCCL_TESTS_REPORT_ARTIFACT_URL, may contain {run_id} and {file}
  std::vector<std::string> artifact_hosts;  // tried in order when no artifact url is set, from the build default or NCCL_TESTS_REPORT_ARTIFACT_HOSTS
  bool artifact_pending = false;  // the artifact endpoint is decided at finalize by resolving artifact_hosts
  std::string artifact_path = "/v1/runs/{run_id}/artifacts/{file}";
  long log_max_time_s = 30;     // NCCL_TESTS_REPORT_LOG_MAX_TIME, per upload
  size_t log_max_bytes = 64u << 20;  // NCCL_TESTS_REPORT_LOG_MAX_MIB, at most 256, larger files are skipped
  size_t log_rows_max_bytes = 1u << 20;  // NCCL_TESTS_REPORT_LOG_ROWS_MAX_KIB, per process when a log goes with the results
};

// envp may be NULL. getenv is used for single lookups either way.
//
// Activation, in order of precedence. NCCL_TESTS_REPORT=0 turns the reporter off.
// NCCL_TESTS_REPORT=1 forces it on. A configured URL turns it on. A pod namespace on the
// skip list turns it off. Inside a cluster with candidate hosts the decision stays pending
// and resolve_default_endpoint picks the first host that resolves. Otherwise it is off.
Config load_config(char **envp);

// Find the artifact endpoint. Returns true and fills artifact_url when one is configured or
// a candidate artifact host resolves.
bool resolve_artifact_endpoint(Config &c);

// Settle a pending configuration. Returns true and fills url when a candidate host
// resolves, or when forced, in which case the first candidate is used unresolved.
bool resolve_default_endpoint(Config &c);

}  // namespace ext
