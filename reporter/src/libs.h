// Versions of the system and of the libraries loaded in this process, collected once at the end
// of the run. Nothing is loaded that was not loaded already, and every lookup is optional.
#pragma once
#include <string>
#include <utility>
#include <vector>

namespace ext {

// (name, value) pairs. Names are stable keys such as driver_version or libibverbs.
std::vector<std::pair<std::string, std::string>> collect_library_versions();

}  // namespace ext
