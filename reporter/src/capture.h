// Keeps a copy of a process's own stdout and stderr, for log upload when the user did not set
// NCCL_DEBUG_FILE. NCCL then writes its debug lines to stdout, and this is the only way to
// keep them without changing what the user sees.
//
// How. For each of fd 1 and fd 2, a pipe replaces the descriptor and a `tee -a FILE` child
// reads the pipe, writes every byte to the original descriptor and appends it to FILE. The
// output keeps its content, order and destination (terminal, mpirun, Slurm or pod log). The
// child is a separate process, so a crash of the test still delivers its last lines: the pipe
// closes, tee drains it and exits. posix_spawnp starts tee without fork handlers, so MPI
// libraries that warn on fork() stay quiet.
#pragma once
#include <sys/types.h>

#include <string>

namespace ext {

struct OutputCapture {
  bool active = false;
  std::string path;            // the file both tee children append to
  int saved[2] = {-1, -1};     // the original fd 1 and fd 2
  pid_t tee_pid[2] = {-1, -1};
};

// Starts copying fd 1 and fd 2 into a new file under dir. Leaves the process unchanged and
// returns false, with a reason in *err, when any step fails, for example when tee is not on
// PATH.
bool capture_start(OutputCapture *c, const std::string &dir, std::string *err);

// Flushes stdio, puts the original descriptors back and waits up to wait_ms for the tee
// children to drain and exit. Safe to call more than once. The file stays until the caller
// removes it.
void capture_stop(OutputCapture *c, int wait_ms);

}  // namespace ext
