// Log files a process can upload as run artifacts, found from settings the user already made.
// Nothing here creates a log or changes where NCCL or the test writes one.
#pragma once
#include <string>
#include <vector>

namespace ext {

// NCCL_DEBUG_FILE with %h, %p and %% expanded the way NCCL expands them. Empty when unset.
std::string nccl_debug_file_path(const char *pattern, const std::string &host, int pid);

// The file given to -J or --output_file on the command line. Empty when there is none.
std::string output_file_from_argv(const std::vector<std::string> &argv);

// A host name reduced to characters that are safe in an artifact file name.
std::string artifact_name_part(const std::string &s);

// Reads the whole file when it exists and is at most max_bytes long. size gets the file size
// whenever the file exists.
bool read_file_capped(const std::string &path, size_t max_bytes, std::string *body, size_t *size);

}  // namespace ext
