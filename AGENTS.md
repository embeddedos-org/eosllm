# AGENTS.md — eosllm

`eosllm` is an on-device LLM inference engine in portable ISO C99: a single
static library (`libeosllm.a`, stable C ABI in `include/eosllm/eosllm.h`) plus
`eosllm-cli`, `eosllm-bench`, `eosllm-convert`, and `eosllm-server` tools, a
VS Code extension (shells out to `eosllm-cli`), and a browser extension (talks
to a local `eosllm-server` on `127.0.0.1:7777`). Default branch: `master`.
(Provenance: `README.md`, `CONTRIBUTING.md`, `Makefile`, `SECURITY.md`,
`run_all_tests.py`, inspected at `master`/`a05d29aa`.)

## Layout

- `include/eosllm/` — public headers (`eosllm.h`, `model.h`, `tensor.h`,
  `quant.h`, `scheduler.h`, `backend.h`, `modality.h`, `os.h`).
- `src/` — engine: `core/`, `format/` (GGUF + native `.eosm`), `kernels/`,
  `quant/`, `sched/`, `tokenizer/`, `modality/`, `os/`, `util/`.
- `tools/` — `eosllm-cli`, `eosllm-server`, `eosllm-bench`, `eosllm-convert`,
  `eosllm-quant-lab` (each ships its own sources; `tools/eosllm-convert/` and
  `tools/eosllm-quant-lab/` carry `pyproject.toml`).
- `tests/` — C unit runner (`tests/unit/test_runner.c`) plus Python suites
  (`tests/unit/`, `tests/functional/`, `tests/performance/`,
  `tests/simulation/`); libFuzzer harnesses and seed corpora in `tests/fuzz/`.
- `docs/` — ABI, architecture, file format, CLI, server, porting, quant
  schemes, threat model; versioned wiki sources live in `docs/wiki/`.
- `browser-extension/` (vite build), `vscode-extension/` (`tsc` compile).

## Build (from `README.md` "Build" and `Makefile`)

Requires a C99 compiler (`cc`/`gcc`/`clang`) and `make`.

```bash
make lib        # build libeosllm.a (default `make all` target)
make tools      # build the CLI, bench, convert, and server binaries
make help       # list all targets
make config     # print resolved EOSLLM_HAVE_* feature flags
```

## Test

C unit suite and checks (from `README.md` "Test" and `Makefile`):

```bash
make test           # build and run the C unit test runner
make smoke          # CLI smoke test (--smoke, no model file needed)
make smoke-all      # smoke variants with output assertions
make server-smoke   # start server, curl /healthz + /caps (POSIX hosts)
make check-errors   # every EOSI_LOG_ERROR string is unique
make check-includes # src/ must not include from tests/, tools/, build/
make sanitize       # ASan/UBSan/LeakSan over the unit suite
make determinism    # unit runner 3x, byte-identical stdout
make all-checks     # full health gate (~5 min; needs no model file)
```

Python suites (entry point `run_all_tests.py`; verified on the pristine tree:
38 passed):

```bash
python -m pytest tests/unit tests/functional tests/performance tests/simulation -v
python3 run_all_tests.py   # same sweep via the repo entry point
```

Extension checks (scripts verified present in each `package.json`; not run
here):

```bash
npm run build    # inside browser-extension/ (vite build)
npm run compile  # inside vscode-extension/ (tsc -p .)
```

Note: `.github/workflows/ci.yml` references branches `main`/`develop` and a
root `requirements.txt`, but the default branch is `master` and there is no
root `requirements.txt`/`pyproject.toml`/`setup.py` — the source tree above
is authoritative.

## Contributing

See `CONTRIBUTING.md` (non-negotiable: C99-only engine, zero hot-path
allocations, no direct OS calls — use the `eos_os_*` shims, vtable-gated
modules, deterministic output; the PR checklist requires `make test` green
and a `CHANGELOG.md` entry). Keep changes scoped, use a fork + feature
branch, link a same-repository issue with `Fixes #N`, and follow
`.github/PULL_REQUEST_TEMPLATE.md`.

## Security

See `SECURITY.md`: only the latest tagged release is supported; report
vulnerabilities via GitHub private vulnerability reporting (preferred) — do
NOT open a public issue. Scope and mitigations are tracked in
`docs/threat_model.md` (unique `eos_last_error()` sites via
`make check-errors`, libFuzzer harnesses in `tests/fuzz/`).

## Existing collaboration model

<!-- generated: eos-ai-scaffold -->
# Agent Responsibilities

Each role owns a slice of the work and does only that slice. Full briefs are in
[.ai/](./.ai/). These are responsibilities, not a required agent count — one
agent may hold several roles on a small change. Split when the roles need
genuinely different context, not by default.

One rule is structural rather than stylistic: **whoever implements does not
approve.** Review is a separate role because self-review reliably misses the
thing the implementer already believes is correct.

## Planner — [.ai/planner.md](./.ai/planner.md)

- Understand the request.
- Break work into tasks.
- Assign work.

## Architect — [.ai/architect.md](./.ai/architect.md)

- Design structure.
- Choose patterns.
- Own dependencies, scalability and maintainability.

## Backend — [.ai/backend.md](./.ai/backend.md)

- APIs
- Database
- Business logic

## Frontend — [.ai/frontend.md](./.ai/frontend.md)

- UI
- Components
- Accessibility

## Testing — [.ai/testing.md](./.ai/testing.md)

- Unit tests
- Integration tests
- Regression tests

## Security — [.ai/security.md](./.ai/security.md)

- Authentication and authorization
- Validation
- Secrets
- Dependency review

## Performance — [.ai/performance.md](./.ai/performance.md)

- Profiling
- Optimization
- Scalability

## Reviewer — [.ai/reviewer.md](./.ai/reviewer.md)

- Final review
- Verify requirements
- Merge findings

## Documentation — [.ai/docs.md](./.ai/docs.md)

- README
- API docs
- Changelog
- Migration and architecture notes

## Release — [.ai/release.md](./.ai/release.md)

- Release notes
- Deployment preparation
- Rollback guidance

---

## Switching roles

Switch when the task changes domain, when specialist knowledge is required,
when independent review is required, or when the context has grown past what
one agent can hold accurately. Every switch runs the protocol in
[HANDOFF.md](./HANDOFF.md).

## Finding work that is not yours

You will. The rule is: **record it, do not absorb it, do not drop it.**

| What you found | Do |
|----------------|-----|
| A defect unrelated to your task | Note it in [TASKS.md](./TASKS.md) and keep going. |
| A defect your change would sit on top of | Stop; say it blocks you; propose fixing it as its own task. |
| A security issue | Report immediately, whatever role you hold. This one never waits for a handoff. |
| A design decision missing from the plan | Return to the architect rather than deciding it inside an implementation. |
| Work that belongs to a role nobody assigned | Say so. An unowned task is how requirements go missing. |

Silently fixing something outside your task makes the diff unreviewable.
Silently ignoring it means nobody ever looks again. Neither is acceptable; the
note is what makes the difference.
