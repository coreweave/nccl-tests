// The reporter implements the nccl-tests extension hooks, buffers what the test
// reports, and at finalize turns it into columnar rows and PUTs them once. With log upload
// enabled, every process delivers its NCCL debug log as an artifact of the same run, or, when no
// artifact endpoint takes it, hands it to rank 0, which sends it to the results endpoint.
//
// Rules, also in the README. Never change the test's stdout, stderr or exit code. Never hang.
// Stay silent unless NCCL_TESTS_REPORT_DEBUG=1. No endpoint means no work at all.
#include "nccl_tests_ext.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
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
const char *kKindLog = "nccl-test-log";
// Log rows travel in their own PUT, which stays under the results endpoint's request limit.
const size_t kLogRowsBudget = 12u << 20;
const size_t kLogChunk = 30000;  // below the 32 KiB string limit of a row

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

// What happened to the NCCL debug log of one process. status is delivered when the artifact
// endpoint took it, in_results when content travels to rank 0 for the results endpoint,
// no_debug_file when NCCL_DEBUG_FILE is unset, missing when the file is not there, and
// over_budget when rank 0 had no room left for it.
struct LogOutcome {
  std::string host, file, status;
  int pid = 0;
  size_t bytes = 0;      // size of the file on disk
  bool truncated = false;
  std::string content;   // only with in_results
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
  bool process_end_called = false;     // the fork gathers log outcomes through process_end
  std::string end_payload;             // what this process hands to rank 0
  std::vector<LogOutcome> logs;        // on the writer, one outcome per process
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
  for (const LogOutcome &l : s.logs) {
    Context c = src("logs");
    ctx_put(c, "log_node", l.host);
    ctx_put(c, "pid", (double)l.pid);
    if (!l.file.empty()) ctx_put(c, "file", l.file);
    ctx_put(c, "bytes", (double)l.bytes);
    if (l.truncated) ctx_put(c, "truncated", std::string("1"));
    add_row(rows, kKindConfig, "log_status", Value::text(l.status), cfg_base, c);
  }

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

// Keeps the first and last halves of text that is over the limit, with a marker between them.
std::string head_and_tail(const std::string &text, size_t max_bytes, bool *truncated) {
  *truncated = text.size() > max_bytes;
  if (!*truncated) return text;
  std::string marker = "\n[... " + std::to_string(text.size() - max_bytes) + " bytes omitted ...]\n";
  size_t keep = max_bytes > marker.size() ? max_bytes - marker.size() : 0;
  return text.substr(0, keep / 2) + marker + text.substr(text.size() - (keep - keep / 2));
}

// Decides what happens to this process's NCCL debug log. The artifact endpoint takes it when one
// is reachable under the shared run id. Otherwise the content goes to rank 0. Only a file the user
// asked NCCL to write is read.
LogOutcome handle_own_log(State &s) {
  LogOutcome o;
  o.host = artifact_name_part(s.node.substr(0, s.node.find('.')));
  o.pid = (int)getpid();
  std::string path = nccl_debug_file_path(getenv("NCCL_DEBUG_FILE"), s.node, o.pid);
  if (path.empty()) { o.status = "no_debug_file"; dbg("no NCCL_DEBUG_FILE, no NCCL log to send"); return o; }
  o.file = "nccl-debug-" + o.host + "-" + std::to_string(o.pid) + ".log";
  std::string body;
  // Read up to the larger of the two limits, so a file too big for one path can still take the other.
  size_t cap = s.cfg.log_max_bytes;
  if (!read_file_capped(path, cap, &body, &o.bytes)) {
    if (o.bytes == 0) { o.status = "missing"; dbg("no readable NCCL log at %s", path.c_str()); return o; }
    // Over the artifact limit, so only the results path with its own limit is left.
    std::ifstream f(path, std::ios::binary);
    body.resize(s.cfg.log_rows_max_bytes);
    f.read(&body[0], (std::streamsize)body.size());
    body.resize((size_t)f.gcount());
    o.content = head_and_tail(body + "\n[... rest of the file omitted ...]\n", s.cfg.log_rows_max_bytes, &o.truncated);
    o.truncated = true;
    o.status = "in_results";
    return o;
  }
  bool id_known = s.writer || s.run_id_shared || !s.cfg.run_id_override.empty();
  bool url_ok = s.cfg.url.empty() || s.cfg.url.find("{run_id}") != std::string::npos || !s.cfg.run_id_override.empty();
  if (id_known && url_ok && resolve_artifact_endpoint(s.cfg)) {
    std::string url = replace_all(replace_all(s.cfg.artifact_url, "{run_id}", s.run_id), "{file}", o.file);
    HttpOptions h;
    h.connect_timeout_s = s.cfg.connect_timeout_s; h.max_time_s = s.cfg.log_max_time_s;
    h.auth_header = s.cfg.auth_header;
    std::string err;
    dbg("PUT %s (%zu bytes)", url.c_str(), body.size());
    if (http_put(url, body, "text/plain", h, &err)) { dbg("artifact %s delivered", o.file.c_str()); o.status = "delivered"; return o; }
    dbg("artifact delivery failed, sending the log with the results instead: %s", err.c_str());
  } else {
    dbg("no artifact endpoint for %s, sending it with the results", o.file.c_str());
  }
  o.content = head_and_tail(body, s.cfg.log_rows_max_bytes, &o.truncated);
  o.status = "in_results";
  return o;
}

// One header line of percent-encoded name=value pairs, then the content.
std::string encode_outcome(const LogOutcome &o) {
  std::string h = "host=" + pct_encode(o.host) + "&pid=" + std::to_string(o.pid) + "&file=" + pct_encode(o.file) +
                  "&status=" + pct_encode(o.status) + "&bytes=" + std::to_string(o.bytes) +
                  "&truncated=" + (o.truncated ? "1" : "0");
  return h + "\n" + o.content;
}

bool decode_outcome(const char *data, size_t len, LogOutcome *o) {
  std::string all(data, len);
  size_t nl = all.find('\n');
  if (nl == std::string::npos) return false;
  for (const auto &kv : decode_local_info(all.substr(0, nl))) {
    if (kv.first == "host") o->host = kv.second;
    else if (kv.first == "pid") o->pid = atoi(kv.second.c_str());
    else if (kv.first == "file") o->file = kv.second;
    else if (kv.first == "status") o->status = kv.second;
    else if (kv.first == "bytes") o->bytes = (size_t)strtoull(kv.second.c_str(), nullptr, 10);
    else if (kv.first == "truncated") o->truncated = kv.second == "1";
  }
  o->content = all.substr(nl + 1);
  return !o->status.empty();
}

// Splits text into row-sized pieces, at a line end where one is near, never inside a UTF-8
// sequence.
std::vector<std::string> chunks_of(const std::string &text) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t n = std::min(kLogChunk, text.size() - pos);
    if (pos + n < text.size()) {
      size_t nl = text.rfind('\n', pos + n - 1);
      if (nl != std::string::npos && nl >= pos + n / 2) n = nl - pos + 1;
      else while (n > 1 && (((unsigned char)text[pos + n]) & 0xC0) == 0x80) --n;
    }
    out.push_back(text.substr(pos, n));
    pos += n;
  }
  return out;
}

// The log rows for every process whose content travelled to rank 0.
std::vector<Row> build_log_rows(const State &s, const timespec &t1) {
  std::vector<Row> rows;
  Context common;
  ctx_put(common, "test_name", s.rc.program.empty() ? (s.argv.empty() ? "" : basename_of(s.argv[0])) : s.rc.program);
  ctx_put(common, "ended_at", iso_utc_ms(t1));
  for (const auto &t : s.cfg.tags) ctx_put(common, "tag_" + t.first, t.second);
  for (const LogOutcome &l : s.logs) {
    if (l.status != "in_results" || l.content.empty()) continue;
    std::vector<std::string> parts = chunks_of(l.content);
    for (size_t i = 0; i < parts.size(); ++i) {
      Context c;
      ctx_put(c, "node", l.host);
      ctx_put(c, "pid", (double)l.pid);
      ctx_put(c, "file", l.file);
      ctx_put(c, "part", (double)i);
      ctx_put(c, "parts", (double)parts.size());
      ctx_put(c, "bytes", (double)l.bytes);
      ctx_put(c, "truncated", std::string(l.truncated ? "1" : "0"));
      add_row(rows, kKindLog, "nccl_debug_log", Value::text(parts[i]), common, c);
    }
  }
  return rows;
}

// Keeps the log content rank 0 sends within one request. Later processes lose their content
// first, and their status says so.
void apply_log_budget(State &s) {
  size_t used = 0;
  for (LogOutcome &l : s.logs) {
    if (l.status != "in_results") continue;
    if (used + l.content.size() > kLogRowsBudget) { l.content.clear(); l.status = "over_budget"; continue; }
    used += l.content.size();
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

void nccl_tests_ext_process_end(const char **data, size_t *len) {
  if (data) *data = nullptr;
  if (len) *len = 0;
  State &s = S();
  if (!s.cfg.active || !data || !len) return;
  s.process_end_called = true;
  if (!s.cfg.logs) return;
  try {
    s.end_payload = encode_outcome(handle_own_log(s));
    *data = s.end_payload.data();
    *len = s.end_payload.size();
  } catch (...) {
    dbg("log handling failed (exception); ignored");
  }
}

void nccl_tests_ext_process_end_gathered(int nprocs, const char *const *data, const size_t *len) {
  State &s = S();
  if (!s.cfg.active || !s.cfg.logs || nprocs <= 0 || !data || !len) return;
  try {
    for (int p = 0; p < nprocs; ++p) {
      LogOutcome o;
      if (data[p] && len[p] && decode_outcome(data[p], len[p], &o)) s.logs.push_back(std::move(o));
    }
  } catch (...) {
    dbg("gathered log outcomes unreadable (exception); ignored");
  }
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
  // A fork without process_end leaves each process to handle its own log here. Only the
  // writer can then send content with the results.
  if (s.cfg.logs && !s.process_end_called) {
    try {
      LogOutcome o = handle_own_log(s);
      if (s.writer) s.logs.push_back(std::move(o));
    } catch (...) {
      dbg("log handling failed (exception); ignored");
    }
  }
  if (!s.writer) { dbg("not the writer rank; no results sent"); return; }
  if (s.sizes_overflow) { dbg("more than %zu result lines; nothing sent", kMaxSizeRecords); return; }
  if (s.cfg.pending) {
    bool ok = resolve_default_endpoint(s.cfg);
    dbg("%s%s", ok ? "endpoint " : "not sent, ", ok ? s.cfg.url.c_str() : s.cfg.reason.c_str());
    if (!ok) return;
  }
  try {
    apply_log_budget(s);
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
    // Logs that no artifact endpoint took go in a PUT of their own kind, so they neither
    // replace the result rows nor count toward their request.
    std::vector<Row> log_rows = build_log_rows(s, t1);
    if (!log_rows.empty()) {
      std::string log_body = rows_to_json(log_rows);
      o.max_time_s = s.cfg.log_max_time_s;
      dbg("PUT %s (%zu log rows, %zu bytes)", url.c_str(), log_rows.size(), log_body.size());
      if (http_put_json(url, log_body, o, &err)) dbg("logs delivered with the results");
      else dbg("log delivery failed (ignored): %s", err.c_str());
    }
  } catch (...) {
    dbg("report failed (exception); ignored");
  }
}

}  // extern "C"
