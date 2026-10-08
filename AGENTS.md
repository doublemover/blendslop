# Repository Guidelines

## Project Structure & Module Organization
The main code lives in `blender_blocking/`. The entry point for the workflow is `blender_blocking/main_integration.py`, with supporting modules grouped by function: `integration/` (image processing, shape matching, Blender ops), `primitives/`, `placement/`, and `shape_matching/`. Tests are Python scripts in `blender_blocking/test_*.py` plus suites like `test_suite_*.py` and the orchestrator `test_runner.py`. Docs are split between `blender_blocking/*.md` and `docs/`, and CI is defined in `.github/workflows/blender-tests.yml`.

## Setup & Dependencies
This project runs inside Blender, so dependencies must be installed into Blender’s bundled Python (not a venv). Use `BLENDER_SETUP.md` and `QUICKSTART.md` for platform-specific steps. A local venv is only for running non-Blender helper scripts like `create_test_images.py`.

## Build, Test, and Development Commands
From `blender_blocking/`:

```bash
# Full Blender test suite (headless)
blender --background --python test_runner.py

# Quick smoke tests
blender --background --python test_runner.py -- --quick
```

Optional local helper workflow (outside Blender):

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python create_test_images.py
```

## Agent & Environment Notes
Automated agents do not have access to Blender in this environment and cannot run Blender at all while working on this repo. Changes must be reasoned about carefully and validated by code inspection. When you pick up changes from an agent, re-run the Blender tests locally before merging.

We work from PowerShell 7.5. For multi-line inputs, prefer here-strings: `@'` ... `'@` (literal) and `@"` ... `"@` (expand variables). The closing marker must be on its own line with no indentation. Easiest quoting: use single quotes for most strings; use double quotes only when you need variable expansion or to embed a single quote.

When referencing `docs/IMPLEMENTATION_SPEC.md`, always include line numbers (e.g., “lines 61-99”) so automation can pull exact context.

## Coding Style & Naming Conventions
Use 4-space indentation and standard Python conventions (PEP 8). Favor `snake_case` for functions/variables, `CapWords` for classes, and keep modules focused by feature area (`integration/`, `primitives/`, etc.). Name new tests as `test_<feature>.py` and wire them into `test_runner.py` if they should run in CI.

## Testing Guidelines
Tests run through the custom Blender test runner (not pytest). Pure-Python suites can run outside Blender (`python blender_blocking/test_runner.py`), but Blender API tests must run in headless Blender (`--background`). Quick mode skips the slow end-to-end validation. Support only the latest official stable Blender release: currently 5.2.2 LTS (verified 2026-10-08). CI provisions that exact official archive on a hosted runner; older releases and previews are unsupported. Keep ordinary API correctness tests.

## Commit & Pull Request Guidelines
Come up with really funny jokes about blenders for commit subjects. Make the details a comprehensive list of the changes. 
