#include "capture.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

namespace ext {

namespace {

void close_fd(int &fd) {
  if (fd >= 0) close(fd);
  fd = -1;
}

// Starts `tee -a path` reading read_end and writing to out_fd. Its own errors go to err_fd.
bool spawn_tee(int read_end, int out_fd, int err_fd, const std::string &path, pid_t *pid, std::string *err) {
  posix_spawn_file_actions_t fa;
  if (posix_spawn_file_actions_init(&fa) != 0) { *err = "posix_spawn_file_actions_init failed"; return false; }
  posix_spawn_file_actions_adddup2(&fa, read_end, 0);
  posix_spawn_file_actions_adddup2(&fa, out_fd, 1);
  posix_spawn_file_actions_adddup2(&fa, err_fd, 2);
  posix_spawnattr_t at;
  posix_spawnattr_init(&at);
  // tee -i ignores SIGINT, so an interrupted test still has its output drained. Every other
  // signal keeps its default, so tee ends with the job.
  sigset_t none;
  sigemptyset(&none);
  posix_spawnattr_setsigmask(&at, &none);
  posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSIGMASK);
  char a0[] = "tee", a1[] = "-a", a2[] = "-i";
  std::string p = path;
  char *argv[] = {a0, a1, a2, &p[0], nullptr};
  int rc = posix_spawnp(pid, "tee", &fa, &at, argv, environ);
  posix_spawn_file_actions_destroy(&fa);
  posix_spawnattr_destroy(&at);
  if (rc != 0) { *err = std::string("cannot start tee: ") + strerror(rc); return false; }
  return true;
}

}  // namespace

bool capture_start(OutputCapture *c, const std::string &dir, std::string *err) {
  if (c->active) return true;
  std::string base = dir.empty() ? "/tmp" : dir;
  c->path = base + "/nccl-tests-output-" + std::to_string((int)getpid()) + ".log";
  int f = open(c->path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (f < 0) { *err = "cannot create " + c->path + ": " + strerror(errno); c->path.clear(); return false; }
  close(f);
  fflush(stdout);
  fflush(stderr);
  int pipes[2][2] = {{-1, -1}, {-1, -1}};
  bool ok = true;
  for (int i = 0; i < 2 && ok; ++i) {
    c->saved[i] = fcntl(i + 1, F_DUPFD_CLOEXEC, 3);
    if (c->saved[i] < 0) { *err = "cannot duplicate fd " + std::to_string(i + 1); ok = false; break; }
    if (pipe(pipes[i]) != 0) { *err = "cannot create a pipe"; ok = false; break; }
    fcntl(pipes[i][0], F_SETFD, FD_CLOEXEC);
    fcntl(pipes[i][1], F_SETFD, FD_CLOEXEC);
  }
  // tee's own error messages go to the original stderr.
  for (int i = 0; i < 2 && ok; ++i)
    ok = spawn_tee(pipes[i][0], c->saved[i], c->saved[1], c->path, &c->tee_pid[i], err);
  if (ok) {
    for (int i = 0; i < 2 && ok; ++i)
      if (dup2(pipes[i][1], i + 1) < 0) { *err = "cannot redirect fd " + std::to_string(i + 1); ok = false; }
  }
  for (int i = 0; i < 2; ++i) { close_fd(pipes[i][0]); close_fd(pipes[i][1]); }
  if (!ok) {
    // Put back whatever was replaced, then let any started tee see EOF and exit.
    for (int i = 0; i < 2; ++i) if (c->saved[i] >= 0) dup2(c->saved[i], i + 1);
    c->active = true;  // so capture_stop reaps the children and closes the saved fds
    capture_stop(c, 1000);
    unlink(c->path.c_str());
    c->path.clear();
    return false;
  }
  c->active = true;
  return true;
}

void capture_stop(OutputCapture *c, int wait_ms) {
  if (!c->active) return;
  c->active = false;
  fflush(stdout);
  fflush(stderr);
  // Pointing fd 1 and fd 2 back at the originals closes the last write end of each pipe, so
  // each tee reads EOF once it has passed everything through.
  for (int i = 0; i < 2; ++i) {
    if (c->saved[i] >= 0) dup2(c->saved[i], i + 1);
  }
  for (int i = 0; i < 2; ++i) {
    if (c->tee_pid[i] <= 0) continue;
    int waited = 0;
    for (;;) {
      int st;
      pid_t r = waitpid(c->tee_pid[i], &st, WNOHANG);
      if (r == c->tee_pid[i] || (r < 0 && errno != EINTR)) break;
      if (waited >= wait_ms) { kill(c->tee_pid[i], SIGTERM); waitpid(c->tee_pid[i], &st, 0); break; }
      struct timespec ts = {0, 10 * 1000 * 1000};
      nanosleep(&ts, nullptr);
      waited += 10;
    }
    c->tee_pid[i] = -1;
  }
  for (int i = 0; i < 2; ++i) close_fd(c->saved[i]);
}

}  // namespace ext
