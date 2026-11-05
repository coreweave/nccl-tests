#include "libs.h"

#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>

#include <cstdio>
#include <fstream>
#include <set>
#include <string>

#include "util.h"

namespace ext {
namespace {

// Basename prefixes of the libraries worth recording. The value is the resolved file name, which
// carries the version for most of them, for example libibverbs.so.1.14.56.0.
const char *kLibraryPrefixes[] = {
    "libibverbs.so", "libmlx5.so", "librdmacm.so", "libnccl.so", "libnccl-net", "libnccl_net",
    "libucp.so", "libucs.so", "libmpi.so", "libhcoll.so", "libsharp.so", "libcudart.so",
    "libcuda.so", "libnvidia-ml.so",
};

std::string library_key(const std::string &base) {
  std::string key = base.substr(0, base.find(".so"));
  for (char &ch : key) if (ch == '-') ch = '_';
  return key;
}

void add(std::vector<std::pair<std::string, std::string>> &out, const std::string &k, const std::string &v) {
  std::string t = trim(v);
  if (!t.empty()) out.emplace_back(k, clip_utf8(t, 200));
}

// Opens a library only if the process already has it loaded.
void *loaded(const char *name) { return dlopen(name, RTLD_NOW | RTLD_NOLOAD); }

void driver_versions(std::vector<std::pair<std::string, std::string>> &out) {
  if (void *h = loaded("libnvidia-ml.so.1")) {
    typedef int (*init_t)(void);
    typedef int (*ver_t)(char *, unsigned int);
    auto init = reinterpret_cast<init_t>(dlsym(h, "nvmlInit_v2"));
    auto ver = reinterpret_cast<ver_t>(dlsym(h, "nvmlSystemGetDriverVersion"));
    auto shut = reinterpret_cast<init_t>(dlsym(h, "nvmlShutdown"));
    char buf[96] = {0};
    if (init && ver && init() == 0) {
      if (ver(buf, sizeof(buf)) == 0) add(out, "driver_version", buf);
      if (shut) shut();
    }
    dlclose(h);
  }
  if (void *h = loaded("libcuda.so.1")) {
    typedef int (*drv_t)(int *);
    auto get = reinterpret_cast<drv_t>(dlsym(h, "cuDriverGetVersion"));
    int v = 0;
    if (get && get(&v) == 0 && v > 0) add(out, "cuda_driver_version", cuda_major_minor(v));
    dlclose(h);
  }
}

void runtime_versions(std::vector<std::pair<std::string, std::string>> &out) {
  typedef int (*mpi_ver_t)(char *, int *);
  if (auto mpi = reinterpret_cast<mpi_ver_t>(dlsym(RTLD_DEFAULT, "MPI_Get_library_version"))) {
    static char buf[8192];
    int len = 0;
    if (mpi(buf, &len) == 0) {
      std::string s(buf, len > 0 && len < (int)sizeof(buf) ? (size_t)len : strnlen(buf, sizeof(buf)));
      add(out, "mpi_library_version", s.substr(0, s.find_first_of("\n,")));
    }
  }
  if (void *h = loaded("libucp.so.0")) {
    typedef const char *(*ucx_ver_t)(void);
    if (auto ucx = reinterpret_cast<ucx_ver_t>(dlsym(h, "ucp_get_version_string"))) {
      if (const char *v = ucx()) add(out, "ucx_version", v);
    }
    dlclose(h);
  }
}

void system_versions(std::vector<std::pair<std::string, std::string>> &out) {
  struct utsname u;
  if (uname(&u) == 0) add(out, "kernel_release", u.release);
  std::ifstream f("/etc/os-release");
  std::string line;
  while (std::getline(f, line)) {
    if (line.compare(0, 12, "PRETTY_NAME=") == 0) {
      std::string v = line.substr(12);
      if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
      add(out, "os_release", v);
    }
  }
}

// Package versions from the dpkg database for the packages that version the RDMA and NCCL stack.
const char *kPackages[] = {"libibverbs1", "ibverbs-providers", "rdma-core", "libnccl2", "doca-ofed"};

void package_versions(std::vector<std::pair<std::string, std::string>> &out) {
  std::ifstream status("/var/lib/dpkg/status");
  std::string line, pkg;
  while (std::getline(status, line)) {
    if (line.compare(0, 9, "Package: ") == 0) {
      pkg = line.substr(9);
    } else if (line.compare(0, 9, "Version: ") == 0 && !pkg.empty()) {
      for (const char *want : kPackages) {
        if (pkg == want) {
          std::string key = "pkg_" + pkg;
          for (char &ch : key) if (ch == '-') ch = '_';
          add(out, key, line.substr(9));
        }
      }
      pkg.clear();
    }
  }
}

// HPC-X records its release in a VERSION file at the top of its install directory.
void hpcx_version(std::vector<std::pair<std::string, std::string>> &out) {
  const char *dir = getenv("HPCX_DIR");
  std::ifstream f(std::string(dir && *dir ? dir : "/opt/hpcx") + "/VERSION");
  std::string first;
  if (std::getline(f, first)) add(out, "hpcx_version", first);
}

void loaded_libraries(std::vector<std::pair<std::string, std::string>> &out) {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  std::set<std::string> seen;
  while (std::getline(maps, line)) {
    size_t slash = line.find('/');
    if (slash == std::string::npos) continue;
    std::string path = line.substr(slash);
    char real[PATH_MAX];
    std::string resolved = realpath(path.c_str(), real) ? std::string(real) : path;
    std::string base = resolved.substr(resolved.rfind('/') + 1);
    if (!seen.insert(base).second) continue;
    for (const char *p : kLibraryPrefixes) {
      if (base.compare(0, strlen(p), p) == 0) { add(out, library_key(base), base); break; }
    }
  }
}

}  // namespace

std::vector<std::pair<std::string, std::string>> collect_library_versions() {
  std::vector<std::pair<std::string, std::string>> out;
  try {
    system_versions(out);
    driver_versions(out);
    runtime_versions(out);
    hpcx_version(out);
    package_versions(out);
    loaded_libraries(out);
  } catch (...) {
  }
  return out;
}

}  // namespace ext
