#!/usr/bin/env bash
# GitHub-hosted Ubuntu only. Download into this job's private runner temp directory.
set -euo pipefail
version="${1:?Blender version required}"
case "$version" in
  5.2.2) series=5.2; checksum=84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168 ;;
  *) printf 'Unsupported CI Blender version: %s\n' "$version" >&2; exit 1 ;;
esac
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]]
: "${RUNNER_TEMP:?GitHub runner temp directory required}"
: "${GITHUB_PATH:?GitHub path file required}"
: "${GITHUB_ENV:?GitHub environment file required}"
archive="blender-$version-linux-x64.tar.xz"
job_dir=$(mktemp -d "$RUNNER_TEMP/blender-$version.XXXXXX")
downloaded=false
for host in https://download.blender.org https://ftp.halifax.rwth-aachen.de/blender; do
  if curl --fail --location --retry 2 --connect-timeout 15 --max-time 240 \
      "$host/release/Blender$series/$archive" --output "$job_dir/$archive"; then
    # A mismatch is a hard failure; never execute or fall back past mismatched bytes.
    printf '%s  %s\n' "$checksum" "$job_dir/$archive" | sha256sum --check --strict
    downloaded=true
    break
  fi
done
[[ "$downloaded" == true ]]
tar -xJf "$job_dir/$archive" -C "$job_dir"
blender_root="$job_dir/blender-$version-linux-x64"
blender_python=$(find "$blender_root/$series/python/bin" -maxdepth 1 \
  -name 'python3.*' -type f -executable -print -quit)
[[ -x "$blender_root/blender" && -n "$blender_python" && -x "$blender_python" ]]
BLENDER_CI_EXPECTED_VERSION="$version" "$blender_root/blender" --background \
  --factory-startup --python-exit-code 1 --python-expr \
  "import bpy, os, sys; expected = tuple(map(int, os.environ['BLENDER_CI_EXPECTED_VERSION'].split('.'))); assert bpy.app.version == expected and bpy.app.version_cycle == 'release', (bpy.app.version, bpy.app.version_cycle, expected); print('Blender:', bpy.app.version_string, 'Python:', sys.version)"
printf '%s\n' "$blender_root" >> "$GITHUB_PATH"
printf 'BLENDER_PYTHON=%s\n' "$blender_python" >> "$GITHUB_ENV"
"$blender_python" --version
