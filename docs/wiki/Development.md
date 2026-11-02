# Development

## Contribution source of truth

[CONTRIBUTING](https://github.com/embeddedos-org/eosllm/blob/master/CONTRIBUTING.md)

Before proposing a change, also review the [README](https://github.com/embeddedos-org/eosllm/blob/master/README.md). Keep changes scoped, add tests appropriate to the affected behavior, and follow the repository's current automation and review requirements.

## Build and dependency inputs found

`Dockerfile`, `Makefile`, `browser-extension/package.json`, `tools/eosllm-convert/pyproject.toml`, `tools/eosllm-quant-lab/pyproject.toml`, `vscode-extension/package.json`.

## Tests found in the default-branch tree

`tests/__init__.py`, `tests/data/.gitignore`, `tests/data/README.md`, `tests/functional/__init__.py`, `tests/functional/test_functional_e2e.py`, `tests/fuzz/.gitignore`, `tests/fuzz/corpus/seed_00.eosm`, `tests/fuzz/corpus/seed_01.eosm`, `tests/fuzz/corpus/seed_02.eosm`, `tests/fuzz/corpus_gguf/seed_00.gguf`, `tests/fuzz/corpus_gguf/seed_01.gguf`, `tests/fuzz/corpus_gguf/seed_02.gguf`, and 15 more.

## Documented test commands

These commands are reproduced from the inspected root README or contributing guide:

```bash
make test           # build and run the unit test runner
```

## Verification baseline

This inventory comes from `master` at [`a05d29aa49d2`](https://github.com/embeddedos-org/eosllm/commit/a05d29aa49d2e2109940b92799b708ef79210606) and found 27 test-related paths among 220 files. Re-check the source tree when that commit is no longer current.
