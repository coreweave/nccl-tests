// Small helpers shared by the reporter for time, ids, host and formatting.
#pragma once
#include <string>
#include <time.h>

namespace ext {

// Monotonic-free wall clock, for started_at/ended_at and the duration.
timespec now_realtime();
double seconds_between(const timespec &a, const timespec &b);
// "2026-10-19T15:04:05.123Z"
std::string iso_utc_ms(const timespec &t);

// Time-ordered random UUID, version 7 of RFC 9562. Falls back to v4-shaped randomness if the
// clock is unusable. Never throws.
std::string uuid_v7();

std::string hostname();

// Nominal size label. 8 -> "8", 33554432 -> "32M", 1.5 GiB -> "1.5G".
std::string humanize_size(unsigned long long n);

// NCCL version code such as 22104 -> "2.21.4". Empty if the code is not plausible.
std::string dotted_nccl_version(int code);

// CUDART_VERSION such as 12040 -> "12.4". Empty if not plausible.
std::string cuda_major_minor(int cudart);

// Trim ASCII whitespace at both ends.
std::string trim(const std::string &s);

// Cut a UTF-8 string to at most max_bytes without splitting a code point.
std::string clip_utf8(const std::string &s, size_t max_bytes);

}  // namespace ext
