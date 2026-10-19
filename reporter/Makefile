# Builds the reporter as a static library for nccl-tests builds, plus a GPU-free test
# harness. From nccl-tests, run make EXT_DIR=/path/to/this/tree.
.RECIPEPREFIX = >
CXX      ?= g++
AR       ?= ar
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++17 -Wall -Wextra -Wno-unused-parameter -fPIC -Iinclude -Isrc
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
> $(CXX) $(CXXFLAGS) $< $(LIB) -ldl -o $@

test: $(BUILDDIR)/fake_run
> bash test/run_tests.sh

clean:
> rm -rf $(BUILDDIR)
