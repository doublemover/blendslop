# Optional Open3D 0.20 setup proposal (not executed)

Retained historical proposal. The final improvement phase authorizes no installation and uses only the already installed Python 3.12/Open3D 0.19 CPU helper. See [final-phase.md](final-phase.md) for executed qualification.

Blender 5.2.2 LTS uses CPython 3.13.13 x64. Its configured module search currently returns `ModuleNotFoundError: No module named 'open3d'`. The repository requirement is **`open3d>=0.19.0`**, not a restrictive 0.19 pin (`blender_blocking/requirements.txt`, line 14). Thus the blocker is an absent local CPython 3.13 installation, not an incompatible supported Python version. The existing `.venv312` contains working Open3D 0.19.0 and is used for isolated CPU qualification without changing Blender.

[Open3D 0.20.0 on PyPI](https://pypi.org/project/open3d/0.20.0/) supplies `open3d-0.20.0-cp313-cp313-win_amd64.whl`, published 16 September 2026, 77.5 MB, SHA256 `689fd4fb3579b2312439b9bceaf92cbb147586b6afc95bdbf6c16e98f5ea15c5`. No source build is needed. [Official release](https://github.com/isl-org/Open3D/releases/tag/v0.20.0).

From the repository root, the smallest staged option: download that exact official wheel to `temp/open3d-py313/wheels`, verify its hash, and install it with the existing Blender Python using `pip --target temp/open3d-py313/site-packages --no-deps`. Add that folder only to the qualifying child process's `sys.path`, after Blender's native libraries. Nothing goes into Program Files, user site, global PATH or registry. Do not expose the Python 3.12 wheel/venv to Blender 3.13. Keep Open3D and LPIPS/Torch in separate children because of the repository's recorded Windows native-runtime conflict.

Upstream's [declared runtime requirements](https://raw.githubusercontent.com/isl-org/Open3D/v0.20.0/python/requirements.txt) are NumPy >=1.18, Dash >=2.6, Werkzeug >=3, Flask >=3, nbformat >=5.7 and ConfigArgParse. Blender already has NumPy 2.3.4. `open3d313-proposal-preflight.json` records whether the other distributions are present. Start with the wheel alone only if those actual imports permit it; otherwise resolve missing distributions and their transitive dependencies to a pinned, wheel-only manifest before seeking installation approval. Full UI/notebook dependencies are not to be guessed or silently downloaded. The core Poisson comparison proceeds with the already installed 0.19 helper in the meantime.

Approval scope, if wanted: official wheel download/hash verification and repository-local installation described above, optional exact manifest of missing dependencies, then isolated import and four-thread capped geometry tests. **No installation, wheel download, source build or system modification has occurred.** Direct Blender integration remains pending this bounded approval; the current CPU helper route has no such blocker.

## Exact optional dependency manifest

All five direct UI/notebook dependencies are absent in Blender. The following pins come from the already-working helper, with Python 3.13.13/Windows marker evaluation and without development or optional extras. NumPy remains Blender-native 2.3.4. Download only official PyPI binary wheels; resolve and hash-check these pins before installation and abort rather than build from source. Native cp313 Windows wheels for MarkupSafe, rpds-py and charset-normalizer are listed on their official PyPI release pages. This manifest has not been installed or downloaded.

- attrs==26.1.0
- blinker==1.9.0
- certifi==2026.4.22
- charset-normalizer==3.4.7
- click==8.3.3
- colorama==0.4.6
- ConfigArgParse==1.7.5
- dash==4.1.0
- fastjsonschema==2.21.2
- Flask==3.1.3
- idna==3.13
- importlib_metadata==9.0.0
- itsdangerous==2.2.0
- Jinja2==3.1.6
- jsonschema==4.26.0
- jsonschema-specifications==2025.9.1
- jupyter_core==5.9.1
- MarkupSafe==3.0.3
- narwhals==2.21.0
- nbformat==5.10.4
- nest-asyncio==1.6.0
- packaging==26.2
- platformdirs==4.9.6
- plotly==6.7.0
- referencing==0.37.0
- requests==2.33.1
- retrying==1.4.2
- rpds-py==0.30.0
- setuptools==81.0.0
- traitlets==5.15.0
- typing_extensions==4.15.0
- urllib3==2.7.0
- Werkzeug==3.1.8
- zipp==3.23.1

Machine-readable closure: temp/improvement-phase-20261006/open3d313-dependency-closure.json. Any source-build requirement, missing wheel or conflict requires a revised proposal. The zero-install .venv312 helper remains the recommended immediate route.
