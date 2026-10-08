# CI/CD Testing Guide for Blender Blocking Tool

This guide explains how to set up continuous integration testing for the Blender Blocking Tool using actual Blender instances in CI/CD pipelines.

## Why Blender Testing in CI/CD is Critical

**Problem**: Mocked tests don't catch real API issues. The boolean solver enum bug (Blender 5.0 removing 'FAST') made it to production because tests didn't run in actual Blender.

**Solution**: Run tests in headless Blender as part of CI/CD pipeline.

## Test Runner

The project includes a comprehensive test runner designed for CI/CD:

```bash
blender --background --python test_runner.py
```

Exit codes:
- `0`: All tests passed
- `1`: One or more tests failed
- `2`: Test runner error (not running in Blender, etc.)

Pure-Python refinement lab coverage is included in the pure phase:

```bash
python blender_blocking/test_runner.py --phase pure --no-progress
```

Full refinement sweeps are intentionally not part of default CI because they generate reports, renders, and per-variant artifacts. Keep those under ignored `temp/refinement-runs/` and upload them only as ad hoc diagnostic artifacts when a workflow explicitly asks for them.

## GitHub Actions Workflow

Create `.github/workflows/blender-tests.yml`:

```yaml
name: Blender Tests

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'blender_blocking/**'
      - '.github/workflows/blender-tests.yml'
  pull_request:
    branches: [ main ]
    paths:
      - 'blender_blocking/**'

jobs:
  test-blender-5:
    name: Test with Blender 5.0
    runs-on: ubuntu-latest
    container:
      image: nytimes/blender:5.0-cpu-ubuntu22.04

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install Python dependencies in Blender
        run: |
          # Find Blender's Python
          BLENDER_PYTHON=$(blender --background --python-expr "import sys; print(sys.executable)" 2>&1 | grep -oP '/[^ ]+python[0-9.]*')
          echo "Blender Python: $BLENDER_PYTHON"

          # Install dependencies
          $BLENDER_PYTHON -m pip install -r blender_blocking/requirements.txt

      - name: Run tests
        run: |
          cd blender_blocking
          blender --background --python test_runner.py

      - name: Upload test artifacts
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: test-logs-blender-5
          path: |
            blender_blocking/*.log
            blender_blocking/test_images/

  test-blender-4:
    name: Test with Blender 4.2
    runs-on: ubuntu-latest
    container:
      image: nytimes/blender:4.2-cpu-ubuntu22.04

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install Python dependencies in Blender
        run: |
          BLENDER_PYTHON=$(blender --background --python-expr "import sys; print(sys.executable)" 2>&1 | grep -oP '/[^ ]+python[0-9.]*')
          echo "Blender Python: $BLENDER_PYTHON"
          $BLENDER_PYTHON -m pip install -r blender_blocking/requirements.txt

      - name: Run tests
        run: |
          cd blender_blocking
          blender --background --python test_runner.py

      - name: Upload test artifacts
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: test-logs-blender-4
          path: |
            blender_blocking/*.log
            blender_blocking/test_images/
```

## GitLab CI

Create `.gitlab-ci.yml`:

```yaml
variables:
  BLENDER_IMAGE: "nytimes/blender:5.0-cpu-ubuntu22.04"

stages:
  - test

test:blender-5:
  stage: test
  image: $BLENDER_IMAGE
  before_script:
    - BLENDER_PYTHON=$(blender --background --python-expr "import sys; print(sys.executable)" 2>&1 | grep -oP '/[^ ]+python[0-9.]*')
    - echo "Installing dependencies in Blender Python $BLENDER_PYTHON"
    - $BLENDER_PYTHON -m pip install -r blender_blocking/requirements.txt
  script:
    - cd blender_blocking
    - blender --background --python test_runner.py
  artifacts:
    when: on_failure
    paths:
      - blender_blocking/*.log
      - blender_blocking/test_images/
  only:
    changes:
      - blender_blocking/**
      - .gitlab-ci.yml

test:blender-4:
  stage: test
  image: "nytimes/blender:4.2-cpu-ubuntu22.04"
  before_script:
    - BLENDER_PYTHON=$(blender --background --python-expr "import sys; print(sys.executable)" 2>&1 | grep -oP '/[^ ]+python[0-9.]*')
    - $BLENDER_PYTHON -m pip install -r blender_blocking/requirements.txt
  script:
    - cd blender_blocking
    - blender --background --python test_runner.py
  artifacts:
    when: on_failure
    paths:
      - blender_blocking/*.log
```

## Docker-Based Testing

### Using Official Blender Docker Images

The New York Times maintains official Blender Docker images:
- https://hub.docker.com/r/nytimes/blender

Available tags:
- `5.0-cpu-ubuntu22.04` - Blender 5.0
- `4.2-cpu-ubuntu22.04` - Blender 4.2
- `4.0-cpu-ubuntu22.04` - Blender 4.0

### Local Docker Testing

Test locally before pushing to CI:

```bash
# Pull Blender image
docker pull nytimes/blender:5.0-cpu-ubuntu22.04

# Run tests in container
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace/blender_blocking \
  nytimes/blender:5.0-cpu-ubuntu22.04 \
  bash -c "
    BLENDER_PYTHON=\$(blender --background --python-expr 'import sys; print(sys.executable)' 2>&1 | grep -oP '/[^ ]+python[0-9.]*')
    \$BLENDER_PYTHON -m pip install -r blender_blocking/requirements.txt
    blender --background --python test_runner.py
  "
```

### Custom Dockerfile

If you need a custom image with dependencies pre-installed:

```dockerfile
FROM nytimes/blender:5.0-cpu-ubuntu22.04

# Install Python dependencies
RUN BLENDER_PYTHON=$(blender --background --python-expr "import sys; print(sys.executable)" 2>&1 | grep -oP '/[^ ]+python[0-9.]*') && \
    $BLENDER_PYTHON -m pip install -r blender_blocking/requirements.txt

# Set working directory
WORKDIR /workspace

# Default command runs tests
CMD ["blender", "--background", "--python", "test_runner.py"]
```

Build and use:
```bash
docker build -t blender-blocking-tests .
docker run --rm -v $(pwd)/blender_blocking:/workspace blender-blocking-tests
```

## Multi-Version Testing Strategy

Test against multiple Blender versions to catch compatibility issues early:

| Blender Version | Python Version | Status | Notes |
|-----------------|----------------|--------|-------|
| 5.0 | 3.11 | Current | Primary target |
| 4.2 | 3.11 | Supported | LTS release |
| 4.0 | 3.10 | Deprecated | Test for legacy support |
| 3.6 | 3.10 | EOL | Drop support |

**Recommendation**: Test against current + previous LTS release minimum.

## Optional Refinement Diagnostics Job

Use this only for manual workflow dispatch or a failing reconstruction branch. It runs one small refinement smoke and uploads the ignored run root for inspection:

```yaml
  refinement-smoke:
    name: Refinement lab smoke
    runs-on: ubuntu-latest
    container:
      image: nytimes/blender:5.0-cpu-ubuntu22.04
    if: github.event_name == 'workflow_dispatch'
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies in Blender Python
        run: |
          BLENDER_PYTHON=$(blender --background --python-expr "import sys; print(sys.executable)" 2>&1 | grep -oP '/[^ ]+python[0-9.]*')
          $BLENDER_PYTHON -m pip install -r blender_blocking/requirements.txt
      - name: Run refinement smoke
        run: |
          blender --background --python blender_blocking/test_e2e_validation.py -- \
            --refinement-suite default-vase \
            --refinement-track profile-loft-refinement \
            --refinement-max-runs 1 \
            --refinement-result-root temp/refinement-runs/ci-smoke-profile \
            --refinement-html-report \
            --no-progress
      - name: Upload refinement artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: refinement-smoke
          path: temp/refinement-runs/ci-smoke-profile
```

## What Tests Must Verify

Critical tests that must run in actual Blender:

### 1. API Compatibility
- Boolean solver enums are valid
- Modifier types and operations work
- bpy.ops commands execute successfully

### 2. Actual Mesh Creation
- Primitives are created with correct geometry
- Boolean operations produce valid meshes
- No degenerate geometry (zero vertices, inside-out faces)

### 3. Scene State
- Objects are added to scene correctly
- No orphan data blocks after cleanup
- Camera and lighting setup works

### 4. Version-Specific Features
- Detect when APIs change between versions
- Graceful degradation for missing features
- Clear error messages for incompatibilities

## Pre-Commit Hooks

Add a pre-commit hook to run quick tests locally:

`.git/hooks/pre-commit`:
```bash
#!/bin/bash

# Quick smoke test before commit
if command -v blender &> /dev/null; then
    echo "Running Blender smoke tests..."
    cd blender_blocking
    blender --background --python test_runner.py -- --quick

    if [ $? -ne 0 ]; then
        echo "❌ Blender tests failed. Commit aborted."
        echo "Fix tests or use 'git commit --no-verify' to skip."
        exit 1
    fi
fi

exit 0
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Troubleshooting CI/CD

### Tests fail in CI but pass locally

**Likely causes:**
1. Different Blender version in CI
2. Missing dependencies in CI environment
3. Different Python version in Blender

**Solution:**
- Lock Blender version in CI config
- Verify dependency installation in CI logs
- Test locally with same Docker image as CI

### Dependency installation fails

**Error**: `No module named 'pip'` in Blender Python

**Solution:**
```bash
# Ensure pip is available
python3 -m ensurepip --default-pip

# Or use system pip with target
pip install --target=/path/to/blender/python/lib -r blender_blocking/requirements.txt
```

### Tests timeout in CI

**Causes:**
- Boolean operations on complex meshes take too long
- Too many slices in test workflow

**Solutions:**
- Use `--quick` mode for CI: `test_runner.py -- --quick`
- Reduce `num_slices` in test cases to 8-10
- Increase CI timeout if needed

### Headless rendering issues

**Error**: `Unable to open a display`

**Solution**: Blender headless mode shouldn't need display. Ensure using `--background` flag:
```bash
blender --background --python test_runner.py
```

## Performance Optimization

### Parallel Testing

Run different test suites in parallel:

```yaml
strategy:
  matrix:
    blender-version: ['5.0', '4.2', '4.0']
    test-suite: ['boolean', 'workflow', 'integration']
```

### Caching

Cache Blender Python dependencies:

```yaml
- name: Cache Blender Python packages
  uses: actions/cache@v4
  with:
    path: ~/.cache/blender/python
    key: blender-${{ matrix.blender-version }}-python-${{ hashFiles('**/requirements.txt') }}
```

### Test Splitting

For large test suites, split across multiple CI jobs:

```yaml
strategy:
  matrix:
    shard: [1, 2, 3, 4]
steps:
  - name: Run test shard
    run: |
      blender --background --python test_runner.py -- --shard ${{ matrix.shard }}/4
```

## Quality Gates

Recommended quality gates for merging:

1. ✅ All tests pass in Blender 5.0
2. ✅ All tests pass in Blender 4.2 (LTS)
3. ✅ No new API deprecation warnings
4. ✅ Code coverage > 80% (measured by Blender test execution)
5. ✅ No memory leaks (orphan data blocks)

## Continuous Monitoring

Set up monitoring for:
- Test execution time trends
- Blender version compatibility
- Dependency security vulnerabilities
- API deprecation warnings

## Further Reading

- [Blender Docker Images](https://hub.docker.com/r/nytimes/blender)
- [Blender Python API Documentation](https://docs.blender.org/api/current/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitLab CI/CD Documentation](https://docs.gitlab.com/ee/ci/)
