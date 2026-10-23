# Builds the reporter as a static library for nccl-tests builds, plus a GPU-free test
# harness. From nccl-tests, run make EXT_DIR=/path/to/this/tree.
.RECIPEPREFIX = >
CXX      ?= g++
AR       ?= ar
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++17 -pthread -Wall -Wextra -Wno-unused-parameter -fPIC -Iinclude -Isrc
# Build-time defaults for the default-on path. Empty here on purpose. The image build passes
# the collector hosts to try, in order, and the pod namespaces that never report.
DEFAULT_HOSTS ?=
SKIP_NAMESPACES ?=
CXXFLAGS += -DNCCL_TESTS_EXT_DEFAULT_HOSTS='"$(DEFAULT_HOSTS)"' -DNCCL_TESTS_EXT_SKIP_NAMESPACES='"$(SKIP_NAMESPACES)"'
BUILDDIR ?= build

SRCS := $(wildcard src/*.cc)
OBJS := $(SRCS:src/%.cc=$(BUILDDIR)/%.o)
LIB  := $(BUILDDIR)/libnccltestsext.a

.PHONY: all lib test clean
all: lib
lib: $(LIB)

$(BUILDDIR):
> mkdir -p $@

$(BUILDDIR)/%.o: src/%.cc include/nccl_tests_ext.h $(wildcard src/*.h) | $(BUILDDIR)
> $(CXX) $(CXXFLAGS) -c $< -o $@

$(LIB): $(OBJS)
> $(AR) rcs $@ $^

$(BUILDDIR)/fake_run: test/fake_run.cc $(LIB)
> $(CXX) $(CXXFLAGS) $< $(LIB) -ldl -pthread -o $@

test: $(BUILDDIR)/fake_run
> bash test/run_tests.sh

clean:
> rm -rf $(BUILDDIR)
