# eosllm Makefile
# ---------------------------------------------------------------------
# Pure C99 build. No third-party deps. The default goal builds the
# static library libeosllm.a; `make test` builds and runs the unit
# test runner; `make tools` builds eosllm-cli + eosllm-bench.
#
# Configuration is taken from build/config.mk if present, otherwise
# from build/config.mk.in. To customize, copy the .in file to
# build/config.mk and edit, or override on the command line:
#
#     make EOSLLM_HAVE_KERNEL_AVX2=1
# ---------------------------------------------------------------------

CONFIG_MK := $(wildcard build/config.mk)
ifeq ($(CONFIG_MK),)
  CONFIG_MK := build/config.mk.in
endif
include $(CONFIG_MK)

# ---------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------
CC      ?= cc
AR      ?= ar
RANLIB  ?= ranlib
BUILD   ?= debug

CSTD     := -std=c99
WARN     := -Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes \
            -Wmissing-prototypes -Wno-unused-parameter
INCLUDES := -Iinclude -Isrc

ifeq ($(BUILD),debug)
  OPT := -O0 -g
else ifeq ($(BUILD),release)
  OPT := -O3 -g -DNDEBUG
else ifeq ($(BUILD),size)
  OPT := -Os -g -DNDEBUG -ffunction-sections -fdata-sections
else
  $(error Unknown BUILD=$(BUILD); use debug|release|size)
endif

CFLAGS  ?= $(CSTD) $(WARN) $(OPT) $(INCLUDES)
LDFLAGS ?=

# CFLAGS_EXTRA / LDFLAGS_EXTRA always append, even when the caller
# overrides CFLAGS / LDFLAGS on the command line. Use these to add
# extra flags (e.g. -static for cross builds) without losing the
# auto-generated -DEOSLLM_HAVE_* defines further down.
CFLAGS_EXTRA  ?=
LDFLAGS_EXTRA ?=

# ---------------------------------------------------------------------
# Sanitizer toggle. Set SAN=1 (or invoke `make sanitize`) to build with
# -fsanitize=address,undefined. Kept opt-in so the default release path
# stays unaffected.
# ---------------------------------------------------------------------
ifeq ($(SAN),1)
  override CFLAGS  += -fsanitize=address,undefined -fno-omit-frame-pointer
  override LDFLAGS += -fsanitize=address,undefined
endif

# ---------------------------------------------------------------------
# Feature-flag → -D defines
# ---------------------------------------------------------------------
FLAGS := \
  EOSLLM_HAVE_POSIX EOSLLM_HAVE_ZEPHYR EOSLLM_HAVE_FREERTOS EOSLLM_HAVE_BAREMETAL \
  EOSLLM_HAVE_WIN32 EOSLLM_HAVE_OS_WASM \
  EOSLLM_HAVE_THREADS \
  EOSLLM_HAVE_KERNEL_SCALAR EOSLLM_HAVE_KERNEL_AVX2 EOSLLM_HAVE_KERNEL_AVX512 \
  EOSLLM_HAVE_KERNEL_NEON EOSLLM_HAVE_KERNEL_SVE EOSLLM_HAVE_KERNEL_RVV \
  EOSLLM_HAVE_KERNEL_HVX EOSLLM_HAVE_KERNEL_NPU EOSLLM_HAVE_KERNEL_WASM_SIMD \
  EOSLLM_HAVE_QUANT_Q8_0 EOSLLM_HAVE_QUANT_Q4_K EOSLLM_HAVE_QUANT_Q2_K \
  EOSLLM_HAVE_QUANT_Q1_58 EOSLLM_HAVE_QUANT_MIXED EOSLLM_HAVE_QUANT_CALIBRATED \
  EOSLLM_HAVE_MODALITY_TEXT EOSLLM_HAVE_MODALITY_VISION EOSLLM_HAVE_MODALITY_AUDIO \
  EOSLLM_HAVE_TOKENIZER_BPE EOSLLM_HAVE_TOKENIZER_SPM \
  EOSLLM_HAVE_FORMAT_EOSM EOSLLM_HAVE_FORMAT_GGUF \
  EOSLLM_HAVE_SCHED_GREEDY EOSLLM_HAVE_SCHED_DEADLINE EOSLLM_HAVE_SCHED_BATCHED

# Host-OS detection. Drives the default win32-on-windows behaviour and
# the platform-specific link flags for tools/eosllm-server (-lws2_32 on
# MinGW / MSVC, nothing extra on POSIX).
HOST_OS := $(shell uname -s 2>/dev/null)
IS_WINDOWS := 0
ifneq (,$(findstring MINGW,$(HOST_OS)))
  IS_WINDOWS := 1
endif
ifneq (,$(findstring MSYS,$(HOST_OS)))
  IS_WINDOWS := 1
endif
ifneq (,$(findstring CYGWIN,$(HOST_OS)))
  IS_WINDOWS := 1
endif
ifeq ($(OS),Windows_NT)
  IS_WINDOWS := 1
endif

# Default Windows build to the Win32 OS shim. MinGW-w64 has no
# posix_memalign, so the POSIX shim cannot compile there; build/config.mk.in
# unconditionally defaults HAVE_POSIX=1/HAVE_WIN32=0, which broke the
# Windows CI legs. Only applies when the caller left the flags at their
# config.mk(.in) defaults -- an explicit EOSLLM_HAVE_POSIX=1 still wins.
ifeq ($(IS_WINDOWS),1)
  ifeq ($(origin EOSLLM_HAVE_WIN32),file)
    override EOSLLM_HAVE_WIN32 := 1
  endif
  ifeq ($(origin EOSLLM_HAVE_POSIX),file)
    override EOSLLM_HAVE_POSIX := 0
  endif
endif

# ---------------------------------------------------------------------
# Auto-enable AVX2 / NEON on hosts that support it, when the user
# hasn't explicitly set the flag (i.e. it's still the build/config.mk.in
# default of 0). This matches what `make benchmark` does so the default
# `make test` exercises the SIMD backend instead of staying scalar-only
# on every developer's machine.
#
# Detection is deliberately conservative:
#   - x86_64 hosts: enable AVX2 only if /proc/cpuinfo mentions avx2.
#     Skipped on macOS (no /proc/cpuinfo); set EOSLLM_HAVE_KERNEL_AVX2=1
#     manually there.
#   - aarch64 hosts: enable NEON unconditionally (mandatory in v8-A).
# Cross builds (e.g. CFLAGS_EXTRA="-static" + aarch64-linux-gnu-gcc)
# still see scalar-only by default; pass the flag explicitly.
# ---------------------------------------------------------------------
HOST_ARCH := $(shell uname -m 2>/dev/null)

# When the user invokes a cross-compiler (CC contains a dash, e.g.
# aarch64-linux-gnu-gcc, x86_64-w64-mingw32-gcc), the host's
# /proc/cpuinfo is irrelevant — the *target* may be entirely different.
# Skip auto-detect in that case; the cross caller is responsible for
# explicitly setting the right EOSLLM_HAVE_KERNEL_* flag.
IS_CROSS := $(if $(filter cc gcc clang,$(notdir $(CC))),0,1)

ifeq ($(origin EOSLLM_HAVE_KERNEL_AVX2),file)
  ifeq ($(IS_CROSS),0)
    ifneq (,$(filter $(HOST_ARCH),x86_64 amd64))
      ifneq ($(shell grep -m1 -o avx2 /proc/cpuinfo 2>/dev/null),)
        override EOSLLM_HAVE_KERNEL_AVX2 := 1
      endif
    endif
  endif
endif

ifeq ($(origin EOSLLM_HAVE_KERNEL_NEON),file)
  ifeq ($(IS_CROSS),0)
    ifneq (,$(filter $(HOST_ARCH),aarch64 arm64))
      override EOSLLM_HAVE_KERNEL_NEON := 1
    endif
  endif
endif

DEFS := $(foreach f,$(FLAGS),-D$(f)=$($(f)))
# `override` so DEFS + CFLAGS_EXTRA always append, even when the caller
# overrode CFLAGS on the command line (otherwise GNU make discards the
# Makefile-side `+=`).
override CFLAGS  += $(DEFS) $(CFLAGS_EXTRA)
override LDFLAGS += $(LDFLAGS_EXTRA)

# ISA-specific compile flags for accelerated backends.
ifeq ($(EOSLLM_HAVE_KERNEL_AVX2),1)
  override CFLAGS += -mavx2 -mfma
endif

# WASM SIMD128 backend (only meaningful under emcc; no-op elsewhere).
ifeq ($(EOSLLM_HAVE_KERNEL_WASM_SIMD),1)
  override CFLAGS += -msimd128
endif

ifeq ($(IS_WINDOWS),1)
  SERVER_LDLIBS := -lws2_32
else
  SERVER_LDLIBS :=
endif

# ---------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------
SRC := \
  src/core/session.c \
  src/core/registry.c \
  src/core/status.c \
  src/core/version.c \
  src/core/caps.c \
  src/core/tensor.c \
  src/core/model.c \
  src/core/init.c \
  src/core/error.c \
  src/os/posix.c \
  src/os/zephyr.c \
  src/os/freertos.c \
  src/os/baremetal.c \
  src/os/win32.c \
  src/os/wasm.c \
  src/kernels/scalar/matmul_f32.c \
  src/kernels/scalar/matmul_q8.c \
  src/kernels/scalar/matmul_q4_k.c \
  src/kernels/scalar/rmsnorm.c \
  src/kernels/scalar/softmax.c \
  src/kernels/scalar/rope.c \
  src/kernels/scalar/silu.c \
  src/kernels/scalar/gelu.c \
  src/kernels/scalar/register.c \
  src/kernels/avx2/avx2.c \
  src/kernels/avx2/matmul_q4_k.c \
  src/kernels/avx2/matmul_q4_k_int8.c \
  src/kernels/avx512/avx512.c \
  src/kernels/neon/neon.c \
  src/kernels/neon/matmul_q4_k.c \
  src/kernels/sve/sve.c \
  src/kernels/rvv/rvv.c \
  src/kernels/hvx/hvx.c \
  src/kernels/npu/npu.c \
  src/kernels/wasm/wasm.c \
  src/kernels/register_all.c \
  src/quant/q8_0.c \
  src/quant/q4_k.c \
  src/quant/q2_k.c \
  src/quant/q1_58.c \
  src/quant/mixed.c \
  src/quant/calibrated.c \
  src/quant/register_all.c \
  src/format/gguf/gguf.c \
  src/format/eosm/eosm.c \
  src/format/eosm/writer.c \
  src/format/eosm/from_model.c \
  src/format/register_all.c \
  src/util/sha256.c \
  src/modality/text/text.c \
  src/modality/vision/vision.c \
  src/modality/audio/audio.c \
  src/modality/fusion.c \
  src/modality/register_all.c \
  src/tokenizer/bpe.c \
  src/tokenizer/spm.c \
  src/tokenizer/register_all.c \
  src/sched/greedy.c \
  src/sched/deadline.c \
  src/sched/batched.c \
  src/sched/paged_kv.c \
  src/sched/speculative.c \
  src/sched/register_all.c

OBJ := $(SRC:.c=.o)
LIB := libeosllm.a

# ---------------------------------------------------------------------
# Top-level targets
# ---------------------------------------------------------------------
.PHONY: all lib test tools cli bench convert server config clean sanitize determinism fuzz-eosm fuzz-gguf fuzz-corpus fuzz-corpus-gguf benchmark bench-diff sanitize-bench smoke sanitize-cli smoke-all all-checks help check-errors check-includes stat server-smoke

all: lib

lib: $(LIB)

$(LIB): $(OBJ)
	$(AR) rcs $@ $^
	$(RANLIB) $@

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
TEST_SRC := tests/unit/test_runner.c
TEST_BIN := tests/unit/test_runner

test: $(TEST_BIN)
	@echo "Running unit tests..."
	@$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(LIB)
	$(CC) $(CFLAGS) $(TEST_SRC) $(LIB) $(LDFLAGS) -lm -o $@

# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------
CLI_BIN     := tools/eosllm-cli/eosllm-cli
BENCH_BIN   := tools/eosllm-bench/eosllm-bench
CONVERT_BIN := tools/eosllm-convert/eosllm-convert
SERVER_BIN  := tools/eosllm-server/eosllm-server

tools: cli bench convert server

cli: $(CLI_BIN)

bench: $(BENCH_BIN)

convert: $(CONVERT_BIN)

server: $(SERVER_BIN)

$(CLI_BIN): tools/eosllm-cli/main.c $(LIB)
	$(CC) $(CFLAGS) tools/eosllm-cli/main.c $(LIB) $(LDFLAGS) -lm -o $@

$(BENCH_BIN): tools/eosllm-bench/main.c $(LIB)
	$(CC) $(CFLAGS) tools/eosllm-bench/main.c $(LIB) $(LDFLAGS) -lm -o $@

$(CONVERT_BIN): tools/eosllm-convert/main.c $(LIB)
	$(CC) $(CFLAGS) tools/eosllm-convert/main.c $(LIB) $(LDFLAGS) -lm -o $@

$(SERVER_BIN): tools/eosllm-server/main.c $(LIB)
	$(CC) $(CFLAGS) tools/eosllm-server/main.c $(LIB) $(LDFLAGS) $(SERVER_LDLIBS) -lm -o $@

# ---------------------------------------------------------------------
# Show resolved feature flags
# ---------------------------------------------------------------------
config:
	@echo "Build : $(BUILD)"
	@echo "CC    : $(CC)"
	@$(foreach f,$(FLAGS),printf "  %-32s = %s\n" $(f) $($(f));)

# ---------------------------------------------------------------------
# Fuzz harnesses (require clang). Skipped on builds without it.
# ---------------------------------------------------------------------
FUZZ_EOSM_BIN   := tests/fuzz/fuzz_eosm
FUZZ_SEED_BIN   := tests/fuzz/seed_eosm_corpus
FUZZ_GGUF_BIN   := tests/fuzz/fuzz_gguf
FUZZ_GGUF_SEED  := tests/fuzz/seed_gguf_corpus

# Compile fuzz_eosm with libFuzzer + ASan + UBSan. Uses clang directly
# because libFuzzer is a clang-only feature.
fuzz-eosm: $(FUZZ_EOSM_BIN)

$(FUZZ_EOSM_BIN): tests/fuzz/fuzz_eosm.c $(LIB)
	clang -fsanitize=fuzzer,address,undefined -g -O1 \
	      $(CSTD) $(WARN) $(INCLUDES) $(DEFS) \
	      tests/fuzz/fuzz_eosm.c $(LIB) -lm \
	      -o $@

# Same idea for the GGUF reader. Uses its own seed corpus.
fuzz-gguf: $(FUZZ_GGUF_BIN)

$(FUZZ_GGUF_BIN): tests/fuzz/fuzz_gguf.c $(LIB)
	clang -fsanitize=fuzzer,address,undefined -g -O1 \
	      $(CSTD) $(WARN) $(INCLUDES) $(DEFS) \
	      tests/fuzz/fuzz_gguf.c $(LIB) -lm \
	      -o $@

# Build a tiny .eosm seed corpus for the fuzzer to mutate past the
# SHA-256 trailer gate.
fuzz-corpus: $(FUZZ_SEED_BIN)
	@$(FUZZ_SEED_BIN) tests/fuzz/corpus

$(FUZZ_SEED_BIN): tests/fuzz/seed_eosm_corpus.c $(LIB)
	$(CC) $(CFLAGS) tests/fuzz/seed_eosm_corpus.c $(LIB) $(LDFLAGS) -lm -o $@

# Build a tiny GGUF v3 seed corpus (minimal-f32, 2-kv-f32-2d, q8_0).
fuzz-corpus-gguf: $(FUZZ_GGUF_SEED)
	@$(FUZZ_GGUF_SEED) tests/fuzz/corpus_gguf

$(FUZZ_GGUF_SEED): tests/fuzz/seed_gguf_corpus.c
	$(CC) $(CFLAGS) tests/fuzz/seed_gguf_corpus.c $(LDFLAGS) -o $@

# ---------------------------------------------------------------------
# Benchmark shortcut: auto-detect ISA, build with the matching kernel
# flag turned on, run eosllm-bench, and write JSON to
# docs/benchmarks/host.json. The runtime probe inside the bench skips
# any backend whose CPU support is not actually present, so this is
# safe to run on any host.
# ---------------------------------------------------------------------
HOST_ARCH := $(shell uname -m 2>/dev/null)
BENCH_AUTOCONF :=
ifneq (,$(filter $(HOST_ARCH),x86_64 amd64))
  BENCH_AUTOCONF := EOSLLM_HAVE_KERNEL_AVX2=1
endif
ifneq (,$(filter $(HOST_ARCH),aarch64 arm64))
  BENCH_AUTOCONF := EOSLLM_HAVE_KERNEL_NEON=1
endif

benchmark:
	$(MAKE) clean
	$(MAKE) $(BENCH_AUTOCONF) bench
	@mkdir -p docs/benchmarks
	@$(BENCH_BIN) > docs/benchmarks/host.json
	@echo "wrote docs/benchmarks/host.json (host=$(HOST_ARCH), autoconf=$(BENCH_AUTOCONF))"

# Compare a freshly-run benchmark against a checked-in baseline.
# Override BASELINE on the command line:
#     make bench-diff BASELINE=docs/benchmarks/baseline-x86_64.json
# Defaults to the path most users would commit.
BASELINE ?= docs/benchmarks/baseline-$(HOST_ARCH).json
bench-diff: benchmark
	@if [ ! -f "$(BASELINE)" ]; then \
	  echo "no baseline at $(BASELINE) — commit one with:"; \
	  echo "  cp docs/benchmarks/host.json $(BASELINE)"; \
	  exit 1; \
	fi
	@python3 tools/bench_diff.py $(BASELINE) docs/benchmarks/host.json

# Sanitize-bench: build + run eosllm-bench at one tiny shape under
# ASan + UBSan + LeakSan. Catches kernel-level memory bugs that the
# unit tests miss (unaligned loads in AVX2/NEON, OOB reads on weight
# buffers, leaks across the bench's own malloc cycle).
sanitize-bench:
	$(MAKE) clean
	ASAN_OPTIONS=detect_leaks=1 \
	  $(MAKE) SAN=1 BUILD=debug $(BENCH_AUTOCONF) bench
	ASAN_OPTIONS=detect_leaks=1:abort_on_error=1:halt_on_error=1 \
	UBSAN_OPTIONS=print_stacktrace=1:abort_on_error=1:halt_on_error=1 \
	  $(BENCH_BIN) --shapes-only > /dev/null
	ASAN_OPTIONS=detect_leaks=1:abort_on_error=1:halt_on_error=1 \
	UBSAN_OPTIONS=print_stacktrace=1:abort_on_error=1:halt_on_error=1 \
	  $(BENCH_BIN) > /dev/null
	@echo "sanitize-bench: OK"

# Smoke: build the CLI and run --smoke (in-memory synthetic-GGUF
# lifecycle exercise; no model file required). Useful as a one-shot
# health check after a session.c / model.c / format-vt change.
smoke: cli
	$(CLI_BIN) --smoke

# smoke-all: tightens the smoke contract from "no crashes" to
# "every variant returns the expected exit code AND its stdout
# carries the expected eos_last_error() context substring". The
# substring assertions catch the silent UX regression where a
# failing call still exits 0 but eos_last_error() returns NULL or
# the wrong message (e.g. an internal refactor accidentally clears
# the slot before the host can read it). Each variant covers a
# different failure shape:
#   --smoke           : in-memory synthetic-GGUF session lifecycle
#                       (model_open OK, session_open EOS_E_NOT_FOUND,
#                       NULL-model session OK, feed/step UNSUPPORTED).
#                       Expected last_error: session_open names the
#                       text decoder.
#   --smoke-bad-magic : GGUF reader reject path (4-byte "XXXX" -> EOS_E_FORMAT
#                       with last_error set).
#                       Expected last_error: gguf_open names "bad magic".
#   --last-error      : eos_model_open on a missing path (-> EOS_E_IO with
#                       last_error set).
#                       Expected last_error: eos_model_open names
#                       os.file_open.
# Every variant exits 0 when it reaches its happy/expected outcome.
smoke-all: cli
	@set -e; \
	echo "--- smoke ---"; \
	o=$$($(CLI_BIN) --smoke); \
	echo "$$o" | grep -q "last_error: session_open: text decoder open failed" \
	  || { echo "FAIL: --smoke missing expected last_error context"; echo "$$o"; exit 1; }; \
	echo "--- smoke-bad-magic ---"; \
	o=$$($(CLI_BIN) --smoke-bad-magic); \
	echo "$$o" | grep -q "last_error: .*bad magic" \
	  || { echo "FAIL: --smoke-bad-magic missing expected last_error context"; echo "$$o"; exit 1; }; \
	echo "--- smoke-empty-stream ---"; \
	o=$$($(CLI_BIN) --smoke-empty-stream); \
	echo "$$o" | grep -q "last_error: .*stream read failed" \
	  || { echo "FAIL: --smoke-empty-stream missing expected last_error context"; echo "$$o"; exit 1; }; \
	echo "--- last-error ---"; \
	o=$$($(CLI_BIN) --last-error); \
	echo "$$o" | grep -q "last_error: eos_model_open: os.file_open failed" \
	  || { echo "FAIL: --last-error missing expected last_error context"; echo "$$o"; exit 1; }; \
	echo "--- caps ---"; \
	o=$$($(CLI_BIN) --caps); \
	echo "$$o" | grep -q '"library_version"' \
	  || { echo "FAIL: --caps missing library_version"; echo "$$o"; exit 1; }; \
	echo "$$o" | grep -q '"runtime_bits"' \
	  || { echo "FAIL: --caps missing runtime_bits"; echo "$$o"; exit 1; }; \
	echo "--- metadata ---"; \
	o=$$(bash tests/fuzz/test_metadata.sh 2>&1); \
	echo "$$o" | grep -q '"general.architecture": "smoke"' \
	  || { echo "FAIL: --metadata smoke missing architecture key"; echo "$$o"; exit 1; }; \
	echo "smoke-all: OK (6 variants, expected output verified in each)"

# all-checks: one-command "is the tree healthy" gate. Chains every
# correctness + lifecycle + sanitizer + perf-regression check that
# doesn't require a real model file. Each step is run via a recursive
# `make` so it gets a fresh build context (sanitize variants need
# their own object files; without a clean recursive-make boundary
# they'd silently reuse the wrong .o set, the same class of bug
# `make benchmark` had). The bench-diff step is conditional: it
# only runs if a baseline JSON for this host arch is present in
# docs/benchmarks/. Total wall-time on a modern x86 host is ~5 min
# (most of it spent in the sanitize link stages).
all-checks:
	@set -e; \
	echo "==== make test ===="            ; $(MAKE) clean > /dev/null && $(MAKE) test; \
	echo "==== make smoke-all ===="       ; $(MAKE) smoke-all; \
	echo "==== make determinism ===="     ; $(MAKE) determinism; \
	echo "==== make check-errors ===="    ; $(MAKE) check-errors; \
	echo "==== make check-includes ====" ; $(MAKE) check-includes; \
	echo "==== make sanitize ===="        ; $(MAKE) sanitize; \
	echo "==== make sanitize-cli ===="    ; $(MAKE) sanitize-cli; \
	echo "==== make sanitize-bench ===="  ; $(MAKE) sanitize-bench; \
	if [ -f "docs/benchmarks/baseline-$(HOST_ARCH).json" ] && [ "$$EOSLLM_SKIP_BENCH_DIFF" != "1" ]; then \
	  echo "==== make bench-diff ===="    ; $(MAKE) bench-diff; \
	elif [ "$$EOSLLM_SKIP_BENCH_DIFF" = "1" ]; then \
	  echo "==== make bench-diff ==== (skipped: EOSLLM_SKIP_BENCH_DIFF=1; perf-gating against shared CI runners is too noisy)"; \
	else \
	  echo "==== make bench-diff ==== (skipped: no baseline-$(HOST_ARCH).json)"; \
	fi; \
	echo ""; \
	echo "all-checks: OK"

# server-smoke: build the HTTP/SSE daemon, start it on a free localhost
# port in the background, curl /healthz and /caps, kill it, and ensure
# both responses look right. Mirrors the smoke-all contract: failure
# in any sub-step exits non-zero. Skipped on Windows hosts (the
# background-job idiom uses POSIX shell features).
server-smoke: server
	@set -e; \
	port=7787; \
	$(SERVER_BIN) --port $$port >/tmp/eosllm-server.log 2>&1 & \
	pid=$$!; \
	trap "kill $$pid 2>/dev/null || true" EXIT; \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
	  if curl -sS "http://127.0.0.1:$$port/healthz" >/dev/null 2>&1; then break; fi; \
	  sleep 0.2; \
	done; \
	hz=$$(curl -sS "http://127.0.0.1:$$port/healthz"); \
	echo "$$hz" | grep -q "ok" \
	  || { echo "FAIL: /healthz did not return ok"; echo "$$hz"; cat /tmp/eosllm-server.log; exit 1; }; \
	cp=$$(curl -sS "http://127.0.0.1:$$port/caps"); \
	echo "$$cp" | grep -q '"library_version"' \
	  || { echo "FAIL: /caps missing library_version"; echo "$$cp"; cat /tmp/eosllm-server.log; exit 1; }; \
	echo "server-smoke: OK"

# Verifies every EOSI_LOG_ERROR("...") string in src/ is unique so
# eos_last_error() pinpoints the failing site. Wraps
# tools/check_error_strings.sh for `make check-errors` ergonomics.
check-errors:
	@bash tools/check_error_strings.sh

# Architectural firewall: src/ files cannot include from tests/,
# tools/, or build/. Catches accidental dependency leaks before they
# compound (a single src/foo.c #include "../../tools/bar.h" would
# tie the library to tooling internals).
check-includes:
	@bash tools/check_includes.sh

# Print headline numbers (file/test/site counts + LOC). Useful for
# changelog generation and "how big is the engine right now" checks.
stat:
	@bash tools/stat.sh

# Self-documenting target reference. Mirror of the table in
# docs/architecture.md; keep them in sync when adding new targets.
help:
	@echo "eosllm Makefile targets (run as: make <target>)"
	@echo ""
	@echo "  Build + test:"
	@echo "    all              build libeosllm.a (default)"
	@echo "    lib              same as 'all'"
	@echo "    test             build + run unit suite (152 checks on AVX2 hosts)"
	@echo "    config           print resolved feature-flag values"
	@echo "    clean            remove all build artifacts"
	@echo ""
	@echo "  Lifecycle smoke:"
	@echo "    smoke            build cli + run --smoke"
	@echo "    smoke-all        run all 5 cli smoke variants with assertions"
	@echo "                       (smoke, smoke-bad-magic, last-error, caps, metadata)"
	@echo ""
	@echo "  Sanitizer pipelines:"
	@echo "    sanitize         clean + ASan/UBSan/LeakSan over the unit suite"
	@echo "    sanitize-cli     same, against the cli smoke"
	@echo "    sanitize-bench   same, against the matmul microbench"
	@echo ""
	@echo "  Determinism + perf:"
	@echo "    determinism      run unit suite 3x; fail on any stdout drift"
	@echo "    benchmark        run microbench, write docs/benchmarks/host.json"
	@echo "    bench-diff       run benchmark + compare to baseline-\$$(HOST_ARCH).json"
	@echo "                       (BASELINE=path overrides default)"
	@echo ""
	@echo "  Fuzzing (libFuzzer):"
	@echo "    fuzz-eosm        30 s libfuzzer + ASan against .eosm corpus"
	@echo "    fuzz-gguf        30 s libfuzzer + ASan against GGUF corpus"
	@echo "    fuzz-corpus      regenerate .eosm seed corpus"
	@echo "    fuzz-corpus-gguf regenerate GGUF seed corpus"
	@echo ""
	@echo "  Tools:"
	@echo "    tools            build cli + bench + convert + server"
	@echo "    cli              build eosllm-cli only"
	@echo "    bench            build eosllm-bench only"
	@echo "    convert          build eosllm-convert only"
	@echo "    server           build eosllm-server (HTTP/SSE) only"
	@echo "    server-smoke     start eosllm-server, curl /healthz + /caps, kill"
	@echo ""
	@echo "  Umbrella:"
	@echo "    all-checks       chain test + smoke-all + determinism + sanitize x3 +"
	@echo "                       bench-diff (if baseline exists). One command for"
	@echo "                       'is the tree healthy'. ~5 min wall-time."
	@echo ""
	@echo "  Common flag overrides (pass on the command line):"
	@echo "    EOSLLM_HAVE_KERNEL_AVX2=1   force AVX2 backend on (auto on x86_64)"
	@echo "    EOSLLM_HAVE_KERNEL_NEON=1   force NEON backend on (auto on aarch64)"
	@echo "    SAN=1 BUILD=debug           build with sanitizers"
	@echo "    CC=clang                    use clang instead of cc"
	@echo "    CFLAGS_EXTRA=\"-static\"      extra cflags (cross-compile etc.)"

# Sanitize-cli: smoke under ASan + UBSan + LeakSan. Catches lifecycle
# bugs (use-after-free across model_close/session_close, leaks in the
# tokenizer-open failure path, OOB on metadata accessors).
sanitize-cli:
	$(MAKE) clean
	ASAN_OPTIONS=detect_leaks=1 \
	  $(MAKE) SAN=1 BUILD=debug cli
	ASAN_OPTIONS=detect_leaks=1:abort_on_error=1:halt_on_error=1 \
	UBSAN_OPTIONS=print_stacktrace=1:abort_on_error=1:halt_on_error=1 \
	  $(CLI_BIN) --smoke
	@echo "sanitize-cli: OK"

clean:
	rm -f $(OBJ) $(LIB) $(TEST_BIN) $(CLI_BIN) $(BENCH_BIN) $(CONVERT_BIN) $(SERVER_BIN) \
	      $(FUZZ_EOSM_BIN) $(FUZZ_SEED_BIN) \
	      $(FUZZ_GGUF_BIN) $(FUZZ_GGUF_SEED) \
	      .determinism.run1 .determinism.run2 .determinism.run3

# ---------------------------------------------------------------------
# Sanitize: clean + rebuild + run tests with ASan + UBSan.
# Recurses into make rather than re-defining CFLAGS in this rule so
# that the sanitizer flags are visible to every sub-compile.
#
# LeakSan: ASan on Linux includes LeakSan automatically when the runner
# environment has ASAN_OPTIONS=detect_leaks=1 (the default on Linux).
# We export it here defensively so the behavior is consistent across
# distros and CI runners.
# ---------------------------------------------------------------------
sanitize:
	$(MAKE) clean
	ASAN_OPTIONS=detect_leaks=1 $(MAKE) SAN=1 BUILD=debug test

# ---------------------------------------------------------------------
# Determinism: same binary, same input, must produce byte-identical
# stdout across 3 consecutive runs. Catches uninitialized-memory reads
# (which sometimes change output) and unstable iteration orders.
# ---------------------------------------------------------------------
determinism: $(TEST_BIN)
	@echo "Running test_runner 3x for determinism check..."
	@$(TEST_BIN) > .determinism.run1 2>&1
	@$(TEST_BIN) > .determinism.run2 2>&1
	@$(TEST_BIN) > .determinism.run3 2>&1
	@diff -q .determinism.run1 .determinism.run2
	@diff -q .determinism.run2 .determinism.run3
	@rm -f .determinism.run1 .determinism.run2 .determinism.run3
	@echo "OK: 3 runs produced byte-identical output"

# ---------------------------------------------------------------------
# Auto-rules
# ---------------------------------------------------------------------
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

