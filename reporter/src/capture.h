// Keeps a copy of a process's own stdout and stderr, for log upload when the user did not set
// NCCL_DEBUG_FILE. NCCL then writes its debug lines to stdout, and this is the only way to
// keep them without changing what the user sees.
//
// How. A pipe replaces fd 1 and fd 2 and a `tee -a -i -- FILE` child reads it, writes every
// byte to the original descriptor and appends it to FILE. When stdout and stderr go to the
// same place (a terminal, or `>log 2>&1`), both share one pipe and one tee, so their order is
// kept exactly. When they go to different places, each gets its own pipe and tee, and only
// the interleaving inside FILE can differ from the order the lines were written. The child is
// a separate process, so a crash of the test still delivers its last lines: the pipe closes,
// tee drains it and exits. posix_spawnp starts tee without fork handlers, so MPI libraries
// that warn on fork() stay quiet.
#pragma once
#include <sys/types.h>

#include <string>

namespace ext {

struct OutputCapture {
  bool active = false;
  std::string path;            // the file the tee children append to
  int saved[2] = {-1, -1};     // the original fd 1 and fd 2
  pid_t tee_pid[2] = {-1, -1};  // one child when stdout and stderr share a destination
};

// Starts copying fd 1 and fd 2 into a new file under dir. Leaves the process unchanged and
// returns false, with a reason in *err, when any step fails, for example when tee is not on
// PATH.
bool capture_start(OutputCapture *c, const std::string &dir, std::string *err);

// Flushes stdio, puts the original descriptors back and waits up to wait_ms for the tee
// children to drain and exit. A child that is still passing output through is never killed;
// it finishes on its own. Returns true when every child exited, so the file is complete.
// Safe to call more than once. The file stays until the caller removes it.
bool capture_stop(OutputCapture *c, int wait_ms);

}  // namespace ext
