# Toolchains

This directory holds cross-compile toolchain fragments used to build
eosllm for non-host targets. Each file is a `.mk` snippet meant to be
included via `make TOOLCHAIN=arm-none-eabi`.

Targets planned for Phase 4:

| File                  | Target                                  |
|-----------------------|-----------------------------------------|
| `arm-none-eabi.mk`    | Cortex-M / Cortex-R bare-metal          |
| `aarch64-linux.mk`    | aarch64 Linux (e.g. for qemu)           |
| `riscv32-elf.mk`      | RV32 bare-metal / Zephyr                |
| `riscv64-linux.mk`    | RV64 Linux (e.g. for qemu)              |
| `hexagon-elf.mk`      | Qualcomm Hexagon DSP                    |

These are placeholders today; the host build (`make`) does not need
any of them.
