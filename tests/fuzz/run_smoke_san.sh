#!/bin/bash
set -u
cd /home/spatchava/embeddedos-org/eosllm
export ASAN_OPTIONS='detect_leaks=1:abort_on_error=1:halt_on_error=1'
export UBSAN_OPTIONS='print_stacktrace=1:abort_on_error=1:halt_on_error=1'
./tools/eosllm-cli/eosllm-cli --smoke
echo "smoke_exit=$?"
