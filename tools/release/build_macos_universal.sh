#!/usr/bin/env bash
# tools/release/build_macos_universal.sh
#
# Build eosllm for arm64 + x86_64 on a macOS runner, then merge into a
# single universal binary via lipo. Called from
# .github/workflows/release.yml::build-macos-universal.
#
# Usage:
#     tools/release/build_macos_universal.sh <staging_dir>
#
# On exit, <staging_dir> contains:
#   bin/eosllm-cli      (universal arm64+x86_64)
#   bin/eosllm-bench    (universal)
#   bin/eosllm-convert  (universal)
#   bin/eosllm-server   (universal)
#   lib/libeosllm.a     (universal)
#   include/eosllm/...
#   share/doc/eosllm/...
#
# This script must run on macos-14 or later (we need lipo + the cc
# wrapper that accepts -target). Apple's "lipo -create -output" merges
# Mach-O objects into a single fat file; the kernel picks the slice
# matching the host CPU at exec time.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <staging_dir>" >&2
    exit 2
fi
STAGE="$1"
shift

mkdir -p "$STAGE/bin" "$STAGE/lib" "$STAGE/include" "$STAGE/share/doc/eosllm"

build_one() {
    local arch="$1"      # arm64 | x86_64
    local target="$2"    # arm64-apple-macos11 | x86_64-apple-macos11
    local outdir="build/$arch"
    mkdir -p "$outdir"

    # Explicitly set BOTH SIMD flags per slice. macos-14 runners are
    # arm64, so when we cross-target x86_64 the Makefile's host-based
    # auto-detect would otherwise wrongly enable NEON; pin both flags
    # to the slice's target ISA.
    local kernel_flags=""
    case "$arch" in
        arm64)
            kernel_flags="EOSLLM_HAVE_KERNEL_NEON=1 EOSLLM_HAVE_KERNEL_AVX2=0" ;;
        x86_64)
            kernel_flags="EOSLLM_HAVE_KERNEL_AVX2=1 EOSLLM_HAVE_KERNEL_NEON=0" ;;
    esac

    make clean >/dev/null

    # cc on macOS is clang; -target overrides the default arch.
    make BUILD=release \
         CC="cc -target $target" \
         $kernel_flags \
         lib tools

    cp libeosllm.a                       "$outdir/libeosllm.a"
    cp tools/eosllm-cli/eosllm-cli       "$outdir/eosllm-cli"
    cp tools/eosllm-bench/eosllm-bench   "$outdir/eosllm-bench"
    cp tools/eosllm-convert/eosllm-convert "$outdir/eosllm-convert" || true
    cp tools/eosllm-server/eosllm-server "$outdir/eosllm-server"
}

build_one arm64  arm64-apple-macos11
build_one x86_64 x86_64-apple-macos11

# lipo each artifact into a universal binary.
for f in eosllm-cli eosllm-bench eosllm-convert eosllm-server; do
    if [[ -f "build/arm64/$f" && -f "build/x86_64/$f" ]]; then
        lipo -create -output "$STAGE/bin/$f" \
             "build/arm64/$f" "build/x86_64/$f"
        chmod +x "$STAGE/bin/$f"
    fi
done

# libeosllm.a is also fat-able via lipo (works on Mach-O archives).
lipo -create -output "$STAGE/lib/libeosllm.a" \
     "build/arm64/libeosllm.a" "build/x86_64/libeosllm.a"

cp -r include/eosllm "$STAGE/include/"
cp LICENSE README.md CHANGELOG.md SECURITY.md CONTRIBUTING.md \
   "$STAGE/share/doc/eosllm/"

echo "build_macos_universal: produced $(ls "$STAGE/bin")"
file "$STAGE/bin/eosllm-cli"
