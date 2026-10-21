// The reporter implements the nccl-tests extension hooks, buffers what the test
// reports, and at finalize turns it into columnar rows and PUTs them once.
//
// Rules, also in the README. Never change the test's stdout, stderr or exit code. Never hang.
// Stay silent unless NCCL_TESTS_REPORT_DEBUG=1. No endpoint means no work at all.
#include "nccl_tests_ext.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "config.h"
#include "http.h"
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
  Half oop, inp;
};
struct Device {
  int rank = 0, group = 0, pid = 0, device = 0;
  std::string hostname, pci, name, serial, uuid;
};
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
  Footer footer;
  std::vector<std::string> errors;
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

  // Context shared by metric rows, in the collector's NCCL vocabulary.
  Context common;
  ctx_put(common, "node", s.node);
  ctx_put(common, "test_name", rc.program.empty() ? (s.argv.empty() ? "" : basename_of(s.argv[0])) : rc.program);
  ctx_put(common, "topology", std::string("collective"));
  ctx_put(common, "nccl_version", s.cfg.nccl_version_env.empty() ? lib : s.cfg.nccl_version_env);
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

  // Per message size, the out-of-place half, as the collector's per-size rows. alg_bw, cpu time
  // and the in-place half are not emitted because the collector does not carry them.
  for (const SizeRec &z : s.sizes) {
    Context shape;
    ctx_put(shape, "message_size", humanize_size(z.nbytes));
    ctx_put(shape, "size_bytes", (double)z.nbytes);
    ctx_put(shape, "count", (double)z.nelem);
    ctx_put(shape, "dtype", z.type);
    ctx_put(shape, "redop", z.op);
    ctx_put(shape, "root", std::to_string(z.root));
    if (z.oop.present) {
      Context c = shape; ctx_put(c, "agg", std::string("sample")); ctx_put(c, "unit", std::string("GB/s"));
      add_row(rows, kKindMetric, "bandwidth", Value::number(z.oop.bus_bw), common, c);
      Context l = shape; ctx_put(l, "agg", std::string("mean")); ctx_put(l, "unit", std::string("us"));
      add_row(rows, kKindMetric, "latency", Value::number(z.oop.time_usec), common, l);
    }
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
  {
    double d = seconds_between(s.t0, t1);
    if (d >= 0) {
      Context c; ctx_put(c, "agg", std::string("sample")); ctx_put(c, "unit", std::string("seconds"));
      add_row(rows, kKindMetric, "duration", Value::number(std::round(d * 1000.0) / 1000.0), common, c);
    }
  }

  // Config rows cover every NCCL_* variable, the test's own settings, the command line and errors.
  Context cfg_base;
  for (const auto &kv : common) if (kv.first != "topology") cfg_base.push_back(kv);
  auto src = [](const char *what) { Context c; ctx_put(c, "source", std::string(what)); return c; };
  for (const auto &kv : s.cfg.nccl_env) add_row(rows, kKindConfig, kv.first, Value::text(kv.second), cfg_base, src("environment"));
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
    add_row(rows, kKindConfig, "error_message", Value::text(clip_utf8(s.errors[i], 200)), cfg_base, src("errors"));

  // One facts row per GPU, in the collector's node-facts shape. The value is the device UUID, else
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
    // facts_base already has node. The per-device node must win, so put it in base position.
    Context base = facts_base;
    for (auto &kv : base) if (kv.first == "node" && !d.hostname.empty()) kv.second = Value::text(d.hostname);
    Context extra;
    for (const auto &kv : c) if (kv.first != "node") extra.push_back(kv);
    add_row(rows, kKindFacts, "gpu", Value::text(id), base, extra);
  }
  return rows;
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
  for (int i = 0; i < argc && argv && argv[i]; ++i) s.argv.push_back(argv[i]);
  if (s.cfg.pending) dbg("active, %s", s.cfg.reason.c_str());
  else dbg("active, %s, %s", s.cfg.reason.c_str(), s.cfg.url.c_str());
}

void nccl_tests_ext_identify_writer(int is_writer) {
  State &s = S();
  if (!s.cfg.active) return;
  if (is_writer) s.writer = true;  // sticky, so a later call with 0 does not clear it
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

void nccl_tests_ext_device(int rank, int group, int pid, const char *hostname, int device,
                           const char *pci_bus_id, const char *device_name, const char *serial,
                           const char *uuid) {
  State &s = S();
  if (!s.cfg.active) return;
  Device d;
  d.rank = rank; d.group = group; d.pid = pid; d.device = device;
  d.hostname = str(hostname); d.pci = str(pci_bus_id); d.name = str(device_name); d.serial = trim(str(serial));
  d.uuid = trim(str(uuid));
  s.devices.push_back(std::move(d));
}

void nccl_tests_ext_size_begin(size_t nbytes, size_t nelem, const char *type_name, const char *op_name, int root) {
  State &s = S();
  if (!s.cfg.active) return;
  SizeRec z;
  z.nbytes = nbytes; z.nelem = nelem; z.type = str(type_name); z.op = str(op_name); z.root = root;
  s.sizes.push_back(std::move(z));
}

void nccl_tests_ext_size_result(int out_of_place, double time_usec, double alg_bw, double bus_bw,
                                int report_errors, long long wrong_elts) {
  State &s = S();
  if (!s.cfg.active || s.sizes.empty()) return;
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
  if (!s.writer) { dbg("not the writer rank; nothing sent"); return; }
  if (s.cfg.pending) {
    bool ok = resolve_default_endpoint(s.cfg);
    dbg("%s%s", ok ? "endpoint " : "not sent, ", ok ? s.cfg.url.c_str() : s.cfg.reason.c_str());
    if (!ok) return;
  }
  try {
    timespec t1 = now_realtime();
    std::vector<Row> rows = build_rows(s, result_code, t1);
    if (rows.empty()) { dbg("no rows; nothing sent"); return; }
    std::string run_id = s.cfg.run_id_override.empty() ? uuid_v7() : s.cfg.run_id_override;
    std::string url = replace_all(s.cfg.url, "{run_id}", run_id);
    std::string body = rows_to_json(rows);
    HttpOptions o;
    o.connect_timeout_s = s.cfg.connect_timeout_s; o.max_time_s = s.cfg.max_time_s;
    o.auth_header = s.cfg.auth_header; o.cert = s.cfg.cert; o.key = s.cfg.key; o.ca = s.cfg.ca;
    std::string err;
    dbg("PUT %s (%zu rows, %zu bytes)", url.c_str(), rows.size(), body.size());
    if (http_put_json(url, body, o, &err)) dbg("delivered");
    else dbg("delivery failed (ignored): %s", err.c_str());
  } catch (...) {
    dbg("report failed (exception); ignored");
  }
}

}  // extern "C"
