// What counts as a credential, shared by the environment filter and the log scrubber.
#pragma once
#include <string>

namespace ext {

// True when a variable name holds a credential or an identity, such as AWS_SECRET_ACCESS_KEY or
// DB_PASSWORD. Names containing PKEY are settings (InfiniBand partition keys), not secrets.
// Case is ignored.
bool is_sensitive_name(const std::string &name);

// True when a value starts like a credential issued by a known service, such as "Bearer " or
// "ghp_".
bool has_secret_value_prefix(const std::string &value);

// What replaces a credential.
extern const char kRedacted[];

// A copy of a log with credentials replaced by kRedacted, in one pass. It replaces private key
// blocks, the rest of an Authorization header, Bearer tokens, tokens that start like a known
// service's, the password in a URL, and the value of any NAME=value whose name is sensitive.
// Everything else is copied byte for byte. redactions, when given, gets the number replaced.
std::string scrub_log(const std::string &in, size_t *redactions);

}  // namespace ext
