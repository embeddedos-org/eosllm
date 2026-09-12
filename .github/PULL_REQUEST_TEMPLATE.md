## Closing issue

<!-- Required for human-authored pull requests. Use a GitHub closing keyword and an issue in THIS repository. Cross-repository links, plain mentions, and placeholders do not satisfy policy. -->

Fixes #<same-repository issue number>

> Replace the placeholder above with a real issue number before requesting review.

## Summary

<!-- Explain what changed and why. For engine changes, name the modules touched in `src/` or `include/eosllm/`. -->

## Validation

<!-- List the commands or manual checks you ran, and their results. -->

- [ ] `make test` passes locally.
- [ ] `python -m pytest tests/unit tests/functional tests/performance tests/simulation -v` passes for Python-suite changes (or `python3 run_all_tests.py`).
- [ ] New direct `malloc`/`free`/`pthread`/`fopen` calls: none in `src/core/`, `src/kernels/`, or `src/modality/` (use the `eos_os_*` shims).
- [ ] New public symbols have entries in `include/eosllm/`; new modules are gated by an `EOSLLM_HAVE_*` flag; new accelerated kernels ship oracle-parity tests under `tests/unit/`.
- [ ] `CHANGELOG.md` updated under `[Unreleased]`.
- [ ] No secrets, credentials, internal hostnames, or production data are included.
