// Runtime configuration of the reporter. Everything comes from the environment or a
// mounted file. Nothing is compiled in. An empty url means the reporter is inactive.
#pragma once
#include <string>
#include <utility>
#include <vector>

namespace ext {

struct Config {
  bool active = false;          // when false, every hook is a no-op with no files and no network
  std::string url;              // may contain {run_id}
  std::string run_id_override;  // NCCL_TESTS_REPORT_RUN_ID
  std::string auth_header;      // NCCL_TESTS_REPORT_AUTH_HEADER, sent verbatim as one header line
  std::string cert, key, ca;    // NCCL_TESTS_REPORT_CERT, _KEY and _CA, for mTLS
  std::string image;            // NCCL_TESTS_REPORT_IMAGE
  long connect_timeout_s = 2;   // NCCL_TESTS_REPORT_CONNECT_TIMEOUT
  long max_time_s = 5;          // NCCL_TESTS_REPORT_MAX_TIME
  bool debug = false;           // NCCL_TESTS_REPORT_DEBUG=1 logs to stderr
  std::vector<std::pair<std::string, std::string>> tags;  // NCCL_TESTS_REPORT_TAG_<NAME> as name and value
  std::vector<std::pair<std::string, std::string>> nccl_env;  // every NCCL_* variable
  std::string slurm_job_id;     // SLURM_JOB_ID
  std::string nccl_version_env; // NCCL_VERSION, if the image sets it
};

// envp may be NULL. getenv is used for single lookups either way.
Config load_config(char **envp);

}  // namespace ext
