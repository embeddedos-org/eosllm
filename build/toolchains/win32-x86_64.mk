# build/toolchains/win32-x86_64.mk — MinGW-w64 cross-compile fragment.
#
# Use as:
#
#     make TOOLCHAIN=win32-x86_64 EOSLLM_HAVE_WIN32=1 \
#          EOSLLM_HAVE_POSIX=0 lib tools
#
# (The release workflow under .github/workflows/release.yml drives this
# from the windows-latest runner via the MSYS2 mingw64 environment;
# CC=x86_64-w64-mingw32-gcc is set there directly. This file mirrors
# the convention reserved by build/toolchains/README.md so that local
# cross-compiles from a Linux host work the same way.)

CC      := x86_64-w64-mingw32-gcc
AR      := x86_64-w64-mingw32-ar
RANLIB  := x86_64-w64-mingw32-ranlib

# Static link by default — Windows release artifacts ship as a single
# .exe each; users won't have the matching MinGW DLLs installed.
CFLAGS_EXTRA  += -DWIN32_LEAN_AND_MEAN
LDFLAGS_EXTRA += -static -static-libgcc -lws2_32

# Force the win32 OS shim on; turn POSIX off (its impl uses
# posix_memalign / clock_gettime which MinGW can fake but we want the
# clean win32 shim instead).
EOSLLM_HAVE_WIN32 := 1
EOSLLM_HAVE_POSIX := 0
