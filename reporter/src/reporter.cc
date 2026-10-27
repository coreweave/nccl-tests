// The reporter implements the nccl-tests extension hooks, buffers what the test
// reports, and at finalize turns it into columnar rows and PUTs them once. With log upload
// enabled, every process also PUTs the log files it can find as artifacts of the same run.
//
// Rules, also in the README. Never change the test's stdout, stderr or exit code. Never hang.
// Stay silent unless NCCL_TESTS_REPORT_DEBUG=1. No endpoint means no work at all.
#include "nccl_tests_ext.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "config.h"
#include "http.h"
#include "libs.h"
#include "logs.h"
#include "rows.h"
#include "util.h"

namespace {
using namespace ext;

const char *kRowVersion = "1.0.0";
const char *kKindMetric = "nccl-test-metric";
const char *kKindConfig = "nccl-test-config";
const char *kKindFacts = "nccl-test-node-facts";

struct Half {
  bool present = false;
  double time_usec = 0, alg_bw = 0, bus_bw = 0;
  bool has_wrong = false;
  long long wrong = 0;
};
struct SizeRec {
  unsigned long long nbytes = 0, nelem = 0;
  std::string type, op;
  int root = 0;
  int cycle = 0;  // index of the -N cycle, counted from 0, inferred from where the sweep restarts
  bool cpu_time = false;
  Half oop, inp;
};

// Result lines kept per run. A run with -N 0 cycles forever and calls the size hooks on every
// cycle, so past this many lines the run stops recording and finalize sends nothing.
static const size_t kMaxSizeRecords = 4096;
struct Device {
  int rank = 0, group = 0, pid = 0, device = 0, gpu_numa = -1;
  std::string hostname, pci, name, serial, uuid, vbios, cpu_affinity, cpu_numa, local_info;
};

// The local info text is name=value pairs joined by '&', with '%', '&', '=' and control and
// space characters percent-encoded, so it survives the line-based gather in one line.
std::string pct_encode(const std::string &s) {
  static const char *hex = "0123456789ABCDEF";
  std::string out;
  for (unsigned char c : s) {
    if (c <= ' ' || c == '%' || c == '&' || c == '=' || c >= 0x7f) { out.push_back('%'); out.push_back(hex[c >> 4]); out.push_back(hex[c & 15]); }
    else out.push_back((char)c);
  }
  return out;
}
std::string pct_decode(const std::string &s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) { out.push_back((char)strtol(s.substr(i + 1, 2).c_str(), nullptr, 16)); i += 2; }
    else out.push_back(s[i]);
  }
  return out;
}
std::vector<std::pair<std::string, std::string>> decode_local_info(const std::string &s) {
  std::vector<std::pair<std::string, std::string>> out;
  size_t start = 0;
  while (start < s.size()) {
    size_t amp = s.find('&', start);
    std::string pair = s.substr(start, amp == std::string::npos ? std::string::npos : amp - start);
    size_t eq = pair.find('=');
    if (eq != std::string::npos && eq > 0) out.emplace_back(pct_decode(pair.substr(0, eq)), pct_decode(pair.substr(eq + 1)));
    if (amp == std::string::npos) break;
    start = amp + 1;
  }
  return out;
}
struct RunConfig {
  bool present = false;
  std::string program, tests_version;
  int headers = 0, library = 0, cudart = 0, nthreads = 0, ngpus = 0;
  unsigned long long min_bytes = 0, max_bytes = 0, step_bytes = 0;
  unsigned long long step_factor = 0;
  int warmup = 0, iters = 0, agg_iters = 0, datacheck = 0, graph = 0, blocking = 0, parallel_init = 0;
};
struct Footer {
  bool present = false;
  int out_of_bounds = 0;
  double avg_bw = 0, check = -1;
  std::string uuid;
};

struct State {
  Config cfg;
  bool initialized = false, writer = false, finalized = false;
  timespec t0{};
  std::vector<std::string> argv;
  std::string node;
  RunConfig rc;
  std::vector<Device> devices;
  std::vector<SizeRec> sizes;
  bool sizes_overflow = false;
  int cycles = 1;
  Footer footer;
  std::vector<std::string> errors;
  std::string run_id;          // minted at init, replaced by the id rank 0 shares
  bool run_id_shared = false;  // rank 0's id arrived through nccl_tests_ext_run_id_shared
  std::vector<std::string> artifacts;  // file names this process delivered as artifacts
};

State &S() {
  static State s;  // constructed on first use, so there is no static-init-order dependence
  return s;
}

void dbg(const char *fmt, ...) {
  if (!S().cfg.debug) return;
  va_list ap;
  va_start(ap, fmt);
  fputs("[nccl-tests-ext] ", stderr);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  va_end(ap);
}

std::string str(const char *s) { return s ? std::string(s) : std::string(); }

std::string basename_of(const std::string &p) {
  size_t i = p.find_last_of('/');
  return i == std::string::npos ? p : p.substr(i + 1);
}

std::string replace_all(std::string s, const std::string &from, const std::string &to) {
  for (size_t p = 0; (p = s.find(from, p)) != std::string::npos; p += to.size()) s.replace(p, from.size(), to);
  return s;
}

void add_row(std::vector<Row> &rows, const char *kind, const std::string &name, const Value &v,
             const Context &base, const Context &extra) {
  Row r;
  if (make_row(r, kind, kRowVersion, name, v, base, extra)) rows.push_back(std::move(r));
}

std::vector<Row> build_rows(const State &s, int result_code, const timespec &t1) {
  std::vector<Row> rows;
  const RunConfig &rc = s.rc;
  std::string lib = dotted_nccl_version(rc.library);

  // Context shared by metric rows.
  Context common;
  ctx_put(common, "node", s.node);
  ctx_put(common, "namespace", s.cfg.pod_namespace);
  ctx_put(common, "test_name", rc.program.empty() ? (s.argv.empty() ? "" : basename_of(s.argv[0])) : rc.program);
  ctx_put(common, "topology", std::string("collective"));
  ctx_put(common, "nccl_version", lib);
  ctx_put(common, "nccl_library_version", lib);
  ctx_put(common, "nccl_headers_version", dotted_nccl_version(rc.headers));
  ctx_put(common, "nccl_tests_version", rc.tests_version);
  ctx_put(common, "cuda_version", cuda_major_minor(rc.cudart));
  ctx_put(common, "image", s.cfg.image);
  ctx_put(common, "job_id", s.cfg.slurm_job_id);
  ctx_put(common, "uuid", s.footer.uuid);
  ctx_put(common, "started_at", iso_utc_ms(s.t0));
  ctx_put(common, "ended_at", iso_utc_ms(t1));
  for (const auto &t : s.cfg.tags) ctx_put(common, "tag_" + t.first, t.second);

  // Per message size, both halves of the result line. The out-of-place half keeps the original
  // names, bandwidth for bus bandwidth and latency for time, and the in-place half and algorithm
  // bandwidth get their own names. Under -C 1 the time is CPU time and is sent as cpu_time.
  for (const SizeRec &z : s.sizes) {
    Context shape;
    ctx_put(shape, "message_size", humanize_size(z.nbytes));
    ctx_put(shape, "size_bytes", (double)z.nbytes);
    ctx_put(shape, "count", (double)z.nelem);
    ctx_put(shape, "dtype", z.type);
    ctx_put(shape, "redop", z.op);
    ctx_put(shape, "root", std::to_string(z.root));
    if (s.cycles > 1) ctx_put(shape, "cycle", (double)z.cycle);
    auto half = [&](const Half &h, const std::string &suffix) {
      if (!h.present) return;
      Context bw = shape; ctx_put(bw, "agg", std::string("sample")); ctx_put(bw, "unit", std::string("GB/s"));
      add_row(rows, kKindMetric, "bandwidth" + suffix, Value::number(h.bus_bw), common, bw);
      add_row(rows, kKindMetric, "alg_bandwidth" + suffix, Value::number(h.alg_bw), common, bw);
      Context tm = shape; ctx_put(tm, "agg", std::string("mean")); ctx_put(tm, "unit", std::string("us"));
      add_row(rows, kKindMetric, (z.cpu_time ? "cpu_time" : "latency") + suffix, Value::number(h.time_usec), common, tm);
    };
    half(z.oop, "");
    half(z.inp, "_in_place");
    if (z.oop.has_wrong || z.inp.has_wrong) {
      long long wrong = (z.oop.has_wrong ? z.oop.wrong : 0) + (z.inp.has_wrong ? z.inp.wrong : 0);
      Context w = shape; ctx_put(w, "agg", std::string("sample")); ctx_put(w, "unit", std::string("count"));
      add_row(rows, kKindMetric, "wrong_elements", Value::number((double)wrong), common, w);
    }
  }

  // Whole-run rows. run() returns a non-zero code on out-of-bounds values or a missed
  // minimum bandwidth, so result_code == 0 and no NCCL error text means PASS.
  const char *verdict = (result_code == 0 && s.errors.empty()) ? "PASS" : "FAIL";
  if (s.footer.present) {
    Context c; ctx_put(c, "agg", std::string("mean")); ctx_put(c, "unit", std::string("GB/s")); ctx_put(c, "verdict", std::string(verdict));
    add_row(rows, kKindMetric, "bandwidth", Value::number(s.footer.avg_bw), common, c);
  }
  {
    Context c; ctx_put(c, "agg", std::string("sample")); ctx_put(c, "unit", std::string("count")); ctx_put(c, "verdict", std::string(verdict));
    add_row(rows, kKindMetric, "returncode", Value::number((double)result_code), common, c);
  }
  if (s.footer.present) {
    Context c; ctx_put(c, "agg", std::string("sample")); ctx_put(c, "unit", std::string("count"));
    add_row(rows, kKindMetric, "out_of_bounds", Value::number((double)s.footer.out_of_bounds), common, c);
    if (s.footer.check >= 0) {
      Context m; ctx_put(m, "agg", std::string("sample")); ctx_put(m, "unit", std::string("GB/s"));
      add_row(rows, kKindMetric, "min_bandwidth_check", Value::number(s.footer.check), common, m);
    }
  }
  if (s.cycles > 1) {
    Context c; ctx_put(c, "agg", std::string("sample")); ctx_put(c, "unit", std::string("count"));
    add_row(rows, kKindMetric, "cycles", Value::number((double)s.cycles), common, c);
  }
  {
    double d = seconds_between(s.t0, t1);
    if (d >= 0) {
      Context c; ctx_put(c, "agg", std::string("sample")); ctx_put(c, "unit", std::string("seconds"));
      add_row(rows, kKindMetric, "duration", Value::number(std::round(d * 1000.0) / 1000.0), common, c);
    }
  }

  // Config rows cover the set variables under the recorded prefixes, the test's own settings,
  // the command line, errors, and the versions of the system and of the libraries loaded in
  // the process.
  Context cfg_base;
  for (const auto &kv : common) if (kv.first != "topology") cfg_base.push_back(kv);
  auto src = [](const char *what) { Context c; ctx_put(c, "source", std::string(what)); return c; };
  for (const auto &kv : s.cfg.recorded_env) add_row(rows, kKindConfig, kv.first, Value::text(kv.second), cfg_base, src("environment"));
  if (rc.present) {
    Context tc = src("test_config");
    add_row(rows, kKindConfig, "nthreads", Value::number(rc.nthreads), cfg_base, tc);
    add_row(rows, kKindConfig, "ngpus", Value::number(rc.ngpus), cfg_base, tc);
    add_row(rows, kKindConfig, "minimum_bytes", Value::number((double)rc.min_bytes), cfg_base, tc);
    add_row(rows, kKindConfig, "maximum_bytes", Value::number((double)rc.max_bytes), cfg_base, tc);
    if (rc.step_factor > 1) add_row(rows, kKindConfig, "step_factor", Value::number((double)rc.step_factor), cfg_base, tc);
    else add_row(rows, kKindConfig, "step_bytes", Value::number((double)rc.step_bytes), cfg_base, tc);
    add_row(rows, kKindConfig, "warmup_iters", Value::number(rc.warmup), cfg_base, tc);
    add_row(rows, kKindConfig, "iterations", Value::number(rc.iters), cfg_base, tc);
    add_row(rows, kKindConfig, "aggregated_iterations", Value::number(rc.agg_iters), cfg_base, tc);
    add_row(rows, kKindConfig, "validation", Value::number(rc.datacheck), cfg_base, tc);
    add_row(rows, kKindConfig, "graph", Value::number(rc.graph), cfg_base, tc);
    add_row(rows, kKindConfig, "blocking_collectives", Value::number(rc.blocking), cfg_base, tc);
    add_row(rows, kKindConfig, "parallel_init", Value::number(rc.parallel_init ? 1 : 0), cfg_base, tc);
  }
  {
    std::string cmd;
    for (size_t i = 0; i < s.argv.size(); ++i) { if (i) cmd.push_back(' '); cmd += s.argv[i]; }
    add_row(rows, kKindConfig, "argv", Value::text(cmd), cfg_base, src("command_line"));
  }
  for (size_t i = 0; i < s.errors.size() && i < 8; ++i)
    add_row(rows, kKindConfig, "error_message", Value::text(clip_utf8(s.errors[i], 4000)), cfg_base, src("errors"));
  for (const auto &kv : collect_library_versions())
    add_row(rows, kKindConfig, kv.first, Value::text(kv.second), cfg_base, src("library"));
  for (const std::string &a : s.artifacts)
    add_row(rows, kKindConfig, "artifact", Value::text(a), cfg_base, src("logs"));

  // One facts row per GPU. The value is the device UUID, else
  // the serial, else the PCI bus id. The serial alone is not enough, because some boards report
  // one serial for the GPUs of a module.
  Context facts_base;
  for (const char *k : {"node", "nccl_version", "nccl_library_version", "cuda_version"})
    for (const auto &kv : common) if (kv.first == k) facts_base.push_back(kv);
  for (const Device &d : s.devices) {
    bool serial_known = !d.serial.empty() && d.serial != "unknown";
    bool uuid_known = !d.uuid.empty() && d.uuid != "unknown";
    std::string id = uuid_known ? d.uuid : (serial_known ? d.serial : d.pci);
    Context c;
    ctx_put(c, "node", d.hostname.empty() ? s.node : d.hostname);
    ctx_put(c, "gpu", (double)d.device);
    if (uuid_known) ctx_put(c, "uuid", d.uuid);
    if (serial_known) ctx_put(c, "serial", d.serial);
    ctx_put(c, "pci_bus_id", d.pci);
    ctx_put(c, "device_name", trim(d.name));
    ctx_put(c, "rank", (double)d.rank);
    ctx_put(c, "group", (double)d.group);
    if (!d.vbios.empty() && d.vbios != "unknown") ctx_put(c, "vbios", d.vbios);
    if (d.gpu_numa >= 0) ctx_put(c, "gpu_numa_node", (double)d.gpu_numa);
    if (!d.cpu_affinity.empty() && d.cpu_affinity != "unknown") ctx_put(c, "cpu_affinity", d.cpu_affinity);
    if (!d.cpu_numa.empty() && d.cpu_numa != "unknown") ctx_put(c, "cpu_numa_nodes", d.cpu_numa);
    // facts_base already has node. The per-device node must win, so put it in base position.
    Context base = facts_base;
    for (auto &kv : base) if (kv.first == "node" && !d.hostname.empty()) kv.second = Value::text(d.hostname);
    Context extra;
    for (const auto &kv : c) if (kv.first != "node") extra.push_back(kv);
    add_row(rows, kKindFacts, "gpu", Value::text(id), base, extra);
  }
  // One row per library version per node, from the local info each process contributed. The
  // first process seen on a node speaks for it.
  std::vector<std::string> nodes_done;
  for (const Device &d : s.devices) {
    std::string host = d.hostname.empty() ? s.node : d.hostname;
    bool done = false;
    for (const auto &n : nodes_done) if (n == host) done = true;
    if (done || d.local_info.empty()) continue;
    nodes_done.push_back(host);
    Context base = facts_base;
    for (auto &kv : base) if (kv.first == "node") kv.second = Value::text(host);
    Context lib; ctx_put(lib, "source", std::string("library"));
    for (const auto &kv : decode_local_info(d.local_info))
      add_row(rows, kKindFacts, kv.first, Value::text(kv.second), base, lib);
  }
  return rows;
}

// PUTs one file as an artifact of the run. Returns true when the endpoint accepted it.
bool put_artifact(State &s, const std::string &path, const std::string &file, const char *content_type) {
  std::string body;
  size_t size = 0;
  if (!read_file_capped(path, s.cfg.log_max_bytes, &body, &size)) {
    if (size > s.cfg.log_max_bytes) dbg("artifact %s skipped, %zu bytes is over the limit", file.c_str(), size);
    else dbg("artifact %s skipped, no readable file at %s", file.c_str(), path.c_str());
    return false;
  }
  std::string url = replace_all(replace_all(s.cfg.artifact_url, "{run_id}", s.run_id), "{file}", file);
  HttpOptions o;
  o.connect_timeout_s = s.cfg.connect_timeout_s; o.max_time_s = s.cfg.log_max_time_s;
  o.auth_header = s.cfg.auth_header;
  std::string err;
  dbg("PUT %s (%zu bytes)", url.c_str(), body.size());
  if (http_put(url, body, content_type, o, &err)) return true;
  dbg("artifact delivery failed (ignored): %s", err.c_str());
  return false;
}

// Uploads the NCCL debug file of this process and, on the writer, the -J output file. Only
// files the user already asked NCCL or the test to write are read.
void upload_logs(State &s) {
  if (!s.writer && !s.run_id_shared && s.cfg.run_id_override.empty()) {
    dbg("logs not sent, this build does not share the run id with this process");
    return;
  }
  // A results URL with a fixed run in its path and no {run_id} names a run this process cannot
  // know, unless NCCL_TESTS_REPORT_RUN_ID gives it.
  if (!s.cfg.url.empty() && s.cfg.url.find("{run_id}") == std::string::npos && s.cfg.run_id_override.empty()) {
    dbg("logs not sent, the results URL has no {run_id}, set NCCL_TESTS_REPORT_RUN_ID to match it");
    return;
  }
  if (!resolve_artifact_endpoint(s.cfg)) { dbg("logs not sent, no artifact endpoint"); return; }
  std::string host = artifact_name_part(s.node.substr(0, s.node.find('.')));
  int pid = (int)getpid();
  std::string debug_path = nccl_debug_file_path(getenv("NCCL_DEBUG_FILE"), s.node, pid);
  if (debug_path.empty()) dbg("no NCCL_DEBUG_FILE, no NCCL log to send");
  else {
    std::string file = "nccl-debug-" + host + "-" + std::to_string(pid) + ".log";
    if (put_artifact(s, debug_path, file, "text/plain")) s.artifacts.push_back(file);
  }
  if (s.writer) {
    std::string out = output_file_from_argv(s.argv);
    if (!out.empty()) {
      std::string file = "nccl-tests-" + host + ".json";
      if (put_artifact(s, out, file, "application/json")) s.artifacts.push_back(file);
    }
  }
}

}  // namespace

extern "C" {

void nccl_tests_ext_init(int argc, char **argv, char **envp) {
  State &s = S();
  if (s.initialized) return;
  s.initialized = true;
  s.cfg = load_config(envp);
  if (!s.cfg.active) {
    // load_config returned before it read the debug switch, so read it here for the one line.
    const char *d = getenv("NCCL_TESTS_REPORT_DEBUG");
    if (d && strcmp(d, "1") == 0) fprintf(stderr, "[nccl-tests-ext] inactive, %s\n", s.cfg.reason.c_str());
    return;
  }
  s.t0 = now_realtime();
  s.node = hostname();
  s.run_id = s.cfg.run_id_override.empty() ? uuid_v7() : s.cfg.run_id_override;
  for (int i = 0; i < argc && argv && argv[i]; ++i) s.argv.push_back(argv[i]);
  if (s.cfg.pending) dbg("active, %s", s.cfg.reason.c_str());
  else dbg("active, %s, %s", s.cfg.reason.c_str(), s.cfg.url.c_str());
}

void nccl_tests_ext_identify_writer(int is_writer) {
  State &s = S();
  if (!s.cfg.active) return;
  if (is_writer) s.writer = true;  // sticky, so a later call with 0 does not clear it
}

void nccl_tests_ext_run_id(char *out, size_t len) {
  if (out == nullptr || len == 0) return;
  out[0] = '\0';
  State &s = S();
  if (!s.cfg.active) return;
  // An id that does not fit is not shared, so no process takes a truncated copy.
  if (s.run_id.size() < len) snprintf(out, len, "%s", s.run_id.c_str());
}

void nccl_tests_ext_run_id_shared(const char *run_id) {
  State &s = S();
  if (!s.cfg.active || run_id == nullptr || *run_id == '\0') return;
  // An explicit NCCL_TESTS_REPORT_RUN_ID is the same on every process and wins.
  if (!s.cfg.run_id_override.empty()) return;
  s.run_id = run_id;
  s.run_id_shared = true;
}

void nccl_tests_ext_run_config(const char *program_name, const char *nccl_tests_version,
                               int nccl_headers_code, int nccl_library_code, int cuda_runtime_version,
                               int nthreads, int ngpus, size_t min_bytes, size_t max_bytes,
                               size_t step_bytes, size_t step_factor, int warmup_iters, int iters,
                               int agg_iters, int datacheck, int cuda_graph_launches,
                               int blocking_coll, int parallel_init) {
  State &s = S();
  if (!s.cfg.active) return;
  RunConfig &r = s.rc;
  r.present = true;
  r.program = basename_of(str(program_name));
  r.tests_version = str(nccl_tests_version);
  r.headers = nccl_headers_code; r.library = nccl_library_code; r.cudart = cuda_runtime_version;
  r.nthreads = nthreads; r.ngpus = ngpus;
  r.min_bytes = min_bytes; r.max_bytes = max_bytes; r.step_bytes = step_bytes; r.step_factor = step_factor;
  r.warmup = warmup_iters; r.iters = iters; r.agg_iters = agg_iters; r.datacheck = datacheck;
  r.graph = cuda_graph_launches; r.blocking = blocking_coll; r.parallel_init = parallel_init;
}

void nccl_tests_ext_local_info(char *out, size_t len) {
  if (out == nullptr || len == 0) return;
  out[0] = '\0';
  State &s = S();
  if (!s.cfg.active) return;
  std::string text;
  try {
    for (const auto &kv : collect_library_versions()) {
      std::string pair = pct_encode(kv.first) + "=" + pct_encode(kv.second);
      if (text.size() + pair.size() + 1 >= len) break;
      if (!text.empty()) text.push_back('&');
      text += pair;
    }
  } catch (...) {
    text.clear();
  }
  snprintf(out, len, "%s", text.c_str());
}

void nccl_tests_ext_device(int rank, int group, int pid, const char *hostname, int device,
                           const char *pci_bus_id, const char *device_name, const char *serial,
                           const char *uuid, const char *vbios, int gpu_numa_node,
                           const char *cpu_affinity, const char *cpu_numa_nodes,
                           const char *local_info) {
  State &s = S();
  if (!s.cfg.active) return;
  Device d;
  d.rank = rank; d.group = group; d.pid = pid; d.device = device;
  d.hostname = str(hostname); d.pci = str(pci_bus_id); d.name = str(device_name); d.serial = trim(str(serial));
  d.uuid = trim(str(uuid));
  d.vbios = trim(str(vbios)); d.gpu_numa = gpu_numa_node;
  d.cpu_affinity = trim(str(cpu_affinity)); d.cpu_numa = trim(str(cpu_numa_nodes));
  d.local_info = str(local_info);
  s.devices.push_back(std::move(d));
}

void nccl_tests_ext_size_begin(size_t nbytes, size_t nelem, const char *type_name, const char *op_name, int root) {
  State &s = S();
  if (!s.cfg.active || s.sizes_overflow) return;
  if (s.sizes.size() >= kMaxSizeRecords) {
    s.sizes_overflow = true;
    s.sizes.clear();
    s.sizes.shrink_to_fit();
    return;
  }
  SizeRec z;
  z.nbytes = nbytes; z.nelem = nelem; z.type = str(type_name); z.op = str(op_name); z.root = root;
  // Each type, op and root gets its own sweep, with the -N cycles inside it. Within one of
  // those sweeps a size that does not grow means the next cycle has started.
  if (!s.sizes.empty()) {
    const SizeRec &prev = s.sizes.back();
    if (prev.type == z.type && prev.op == z.op && prev.root == z.root)
      z.cycle = nbytes <= prev.nbytes ? prev.cycle + 1 : prev.cycle;
  }
  if (z.cycle + 1 > s.cycles) s.cycles = z.cycle + 1;
  s.sizes.push_back(std::move(z));
}

void nccl_tests_ext_size_result(int out_of_place, double time_usec, double alg_bw, double bus_bw,
                                int report_errors, long long wrong_elts, int cpu_time) {
  State &s = S();
  if (!s.cfg.active || s.sizes_overflow || s.sizes.empty()) return;
  if (cpu_time) s.sizes.back().cpu_time = true;
  Half &h = out_of_place ? s.sizes.back().oop : s.sizes.back().inp;
  h.present = true;
  h.time_usec = time_usec; h.alg_bw = alg_bw; h.bus_bw = bus_bw;
  h.has_wrong = report_errors != 0; h.wrong = wrong_elts;
}

void nccl_tests_ext_footer(int out_of_bounds, double avg_bus_bw, double check_avg_bw, const char *uuid) {
  State &s = S();
  if (!s.cfg.active) return;
  s.footer.present = true;
  s.footer.out_of_bounds = out_of_bounds; s.footer.avg_bw = avg_bus_bw; s.footer.check = check_avg_bw;
  s.footer.uuid = str(uuid);
}

void nccl_tests_ext_error(const char *message) {
  State &s = S();
  if (!s.cfg.active) return;
  std::string m = trim(str(message));
  if (!m.empty()) s.errors.push_back(m);
}

void nccl_tests_ext_finalize(int result_code) {
  State &s = S();
  if (!s.initialized || !s.cfg.active || s.finalized) return;
  s.finalized = true;
  // The run ends here, so log uploads do not count toward the reported duration.
  timespec t1 = now_realtime();
  if (s.cfg.logs) {
    try { upload_logs(s); } catch (...) { dbg("log upload failed (exception); ignored"); }
  }
  if (!s.writer) { dbg("not the writer rank; no results sent"); return; }
  if (s.sizes_overflow) { dbg("more than %zu result lines; nothing sent", kMaxSizeRecords); return; }
  if (s.cfg.pending) {
    bool ok = resolve_default_endpoint(s.cfg);
    dbg("%s%s", ok ? "endpoint " : "not sent, ", ok ? s.cfg.url.c_str() : s.cfg.reason.c_str());
    if (!ok) return;
  }
  try {
    std::vector<Row> rows = build_rows(s, result_code, t1);
    if (rows.empty()) { dbg("no rows; nothing sent"); return; }
    std::string url = replace_all(s.cfg.url, "{run_id}", s.run_id);
    std::string body = rows_to_json(rows);
    HttpOptions o;
    o.connect_timeout_s = s.cfg.connect_timeout_s; o.max_time_s = s.cfg.max_time_s;
    o.auth_header = s.cfg.auth_header;
    std::string err;
    dbg("PUT %s (%zu rows, %zu bytes)", url.c_str(), rows.size(), body.size());
    if (http_put_json(url, body, o, &err)) dbg("delivered");
    else dbg("delivery failed (ignored): %s", err.c_str());
  } catch (...) {
    dbg("report failed (exception); ignored");
  }
}

}  // extern "C"
