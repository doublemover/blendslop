#!/usr/bin/env python3
"""Run an explicit bounded argv in fresh owned evidence; never release unknown joins."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender_blocking"))
from utils.owned_process_supervisor import run_bounded_process
from utils.run_ownership import OwnedRun

MAX_BYTES = 8 * 1024 ** 3
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _supports_complete_tree():
    return os.name == "nt"


def _file_record(path, *, max_bytes):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("provenance input is not a bounded regular file: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if before.st_size > max_bytes:
            raise ValueError("provenance input exceeds its read bound")
        read_bytes = 0
        while True:
            chunk = handle.read(min(1048576, max_bytes - read_bytes + 1))
            if not chunk:
                break
            read_bytes += len(chunk)
            if read_bytes > max_bytes:
                raise ValueError("provenance input grew beyond its read bound")
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    current = path.stat()
    signature = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if signature(before) != signature(after) or signature(after) != signature(current):
        raise ValueError("provenance input changed or was replaced during hashing: " + str(path))
    return {"path": str(path), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
            "sha256": digest.hexdigest()}


def prepare_command(command, *, output_parent, timeout_s, max_memory_bytes,
                    max_rss_bytes, threads, join_timeout_s=5., cwd=None,
                    inputs=(), portable_primary_only=False):
    """Validate arguments and read provenance before ownership creation or launch."""
    if (not isinstance(command, (tuple, list)) or not 1 <= len(command) <= 128 or
            any(not isinstance(arg, (str, os.PathLike)) for arg in command)):
        raise ValueError("an explicit bounded executable argv is required")
    original = [os.fspath(arg) for arg in command]
    if (not original[0] or any("\x00" in arg for arg in original) or
            sum(len(arg.encode("utf8")) for arg in original) > 65536):
        raise ValueError("command argv is empty, contains NUL, or exceeds its byte bound")
    for value in (timeout_s, join_timeout_s):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value <= 0):
            raise ValueError("positive finite work and join timeouts are required")
    if join_timeout_s > 5 or timeout_s + join_timeout_s > 90:
        raise ValueError("work plus final joins must fit 90 seconds; joins at most 5 seconds")
    for value in (max_memory_bytes, max_rss_bytes):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or
                                  not 0 < value <= MAX_BYTES):
            raise ValueError("declared memory/RSS limits must be positive integers at most 8 GiB")
    if max_memory_bytes is None:
        raise ValueError("an explicit committed-memory bound is required")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads not in (1, 2):
        raise ValueError("an explicit thread control of one or two is required")
    if not isinstance(portable_primary_only, bool):
        raise ValueError("portable-primary-only must be a boolean")
    if portable_primary_only:
        if max_rss_bytes is not None:
            raise ValueError("portable primary-only scope has no qualified tree RSS monitor")
    elif not _supports_complete_tree() or max_rss_bytes is None:
        raise ValueError("complete-tree mode requires Windows and an explicit RSS bound")
    working = Path(cwd or Path.cwd()).resolve(strict=True)
    if not working.is_dir():
        raise ValueError("command working directory must already exist")
    parent = Path(output_parent).absolute()
    if (parent != parent.resolve() or parent.is_symlink() or
            getattr(parent, "is_junction", lambda: False)() or
            (parent.exists() and not parent.is_dir())):
        raise ValueError("output parent must be a plain directory path")
    supplied = Path(original[0])
    if supplied.is_absolute() or any(separator in original[0] for separator in ("/", "\\")):
        executable = supplied if supplied.is_absolute() else working / supplied
    else:
        located = shutil.which(original[0])
        if located is None:
            raise ValueError("explicit executable was not found")
        executable = Path(located)
    executable = executable.resolve(strict=True)
    if (not executable.is_file() or
            (os.name == "nt" and executable.suffix.lower() not in (".exe", ".com")) or
            (os.name != "nt" and not os.access(executable, os.X_OK))):
        raise ValueError("command executable must be an existing native executable")
    actual = [str(executable), *original[1:]]
    if os.name == "nt":
        import subprocess
        if len(subprocess.list2cmdline(actual).encode("utf-16-le")) // 2 > 32766:
            raise ValueError("command exceeds the Windows command-line bound")
    if executable.stem.lower() == "blender":
        native = actual[1:actual.index("--") if "--" in actual else len(actual)]
        indices = [i for i, arg in enumerate(native)
                   if arg in ("--threads", "-t") or arg.startswith("--threads=")
                   or (arg.startswith("-t") and arg[2:].isdigit())]
        if (len(indices) != 1 or native[indices[0]] not in ("--threads", "-t") or
                indices[0] + 1 >= len(native) or native[indices[0] + 1] != str(threads) or
                not any(arg in ("--background", "-b") for arg in native)):
            raise ValueError("Blender requires background mode and one explicit matching --threads/-t control")
    if not isinstance(inputs, (tuple, list)) or len(inputs) > 64:
        raise ValueError("at most 64 explicit provenance inputs are supported")
    input_records = []
    remaining = 268435456
    for path in inputs:
        row = _file_record(path, max_bytes=min(67108864, remaining))
        input_records.append(row)
        remaining -= row["bytes"]
    sources = [Path(__file__).resolve(), ROOT / "blender_blocking/utils/owned_process_supervisor.py",
               ROOT / "blender_blocking/utils/run_ownership.py"]
    provenance = {"protocol": "bounded-owned-command-provenance-v1",
                  "executable": _file_record(executable, max_bytes=536870912),
                  "sources": [_file_record(path, max_bytes=1048576) for path in sources],
                  "explicit_shared_inputs": input_records,
                  "scope": "read-only current byte snapshots; no historical ownership or input locking"}
    command_record = {"protocol": "bounded-owned-command-v1", "requested_argv": original,
                      "actual_argv": actual, "cwd": str(working), "timeout_s": timeout_s,
                      "join_timeout_s": join_timeout_s, "total_supervision_budget_s": timeout_s + join_timeout_s,
                      "max_job_committed_bytes": max_memory_bytes, "max_rss_bytes": max_rss_bytes,
                      "threads": threads, "thread_environment": {key: str(threads) for key in THREAD_VARIABLES},
                      "thread_scope": "explicit Blender option and numerical-library environment controls; no kernel thread quota",
                      "portable_primary_only": portable_primary_only,
                      "child_output_scope": "child artifacts remain with their own producer; no path substitution or adoption"}
    return parent, command_record, provenance


def _write_owned_json(owner, name, value):
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf8")
    owner.reserve_bytes(len(encoded))
    with (owner.root / name).open("xb") as handle:
        handle.write(encoded)
    owner.register_file(name, "diagnostic")


def run_owned_command(command, **options):
    """Retain evidence and close only a complete-tree-qualified fresh lease."""
    parent, declaration, provenance = prepare_command(command, **options)
    owner = OwnedRun(parent, producer="bounded_owned_command", max_generated_bytes=8388608,
                     shared_inputs={row["path"]: row["sha256"] for row in provenance["explicit_shared_inputs"]})
    resource = None
    complete = False
    errors = []
    try:
        _write_owned_json(owner, "command.json", declaration)
        _write_owned_json(owner, "provenance.json", provenance)
        environment = dict(os.environ)
        environment.update(declaration["thread_environment"])
        resource = run_bounded_process(
            declaration["actual_argv"], log_path=owner.root / "stdout.log",
            timeout_s=declaration["timeout_s"], join_timeout_s=declaration["join_timeout_s"],
            max_memory_bytes=declaration["max_job_committed_bytes"], max_rss_bytes=declaration["max_rss_bytes"],
            require_complete_tree=not declaration["portable_primary_only"],
            cwd=declaration["cwd"], env=environment)
        complete = (not declaration["portable_primary_only"] and
                    resource.get("lifecycle_complete") is True)
        success = (complete and resource.get("status") == "succeeded" and
                   type(resource.get("returncode")) is int and resource["returncode"] == 0)
        if not success:
            owner.mark_failed("primary command failed" if complete else "complete-tree lifecycle remains unconfirmed")
        _write_owned_json(owner, "resource-receipt.json", resource)
        log = owner.root / "stdout.log"
        if log.is_file():
            owner.register_file("stdout.log", "diagnostic")
        _write_owned_json(owner, "run-result.json", {
            "protocol": "bounded-owned-command-result-v1", "status": "succeeded" if success else "failed",
            "primary_returncode": resource.get("returncode"), "lifecycle_complete": complete,
            "lease_release_decision": "permitted_complete_tree" if complete else "blocked_unconfirmed_tree",
            "lease_status_at_receipt": "active", "run_id": owner.run_id,
            "stdout_status": "retained" if log.is_file() else "unavailable; supervisor did not create a log",
            "portable_primary_only": declaration["portable_primary_only"]})
        if complete:
            owner.close(state="succeeded" if success else "failed")
    except Exception as exc:
        errors.append(repr(exc))
        owner.mark_failed("orchestration failed: " + repr(exc))
        try:
            if resource is None:
                _write_owned_json(owner, "resource-receipt.json", {
                    "protocol": "bounded-owned-command-resource-unavailable-v1",
                    "status": "failed", "returncode": None, "lifecycle_complete": False,
                    "resource_observations": "unavailable; supervisor returned no receipt",
                    "error": repr(exc), "receipt_source": "wrapper exception history"})
            _write_owned_json(owner, "orchestration-error.json", {
                "error": repr(exc), "lifecycle_complete": complete,
                "resource_receipt_available": resource is not None,
                "resource_receipt": resource,
                "lease_release_decision": "permitted_complete_tree" if complete else "blocked_unconfirmed_tree"})
            if (owner.root / "stdout.log").is_file():
                owner.register_file("stdout.log", "diagnostic")
            if complete:
                owner.close(state="failed")
        except Exception as recording_error:
            errors.append(repr(recording_error))
    primary_code = resource.get("returncode") if isinstance(resource, dict) else None
    succeeded = owner.closed and owner.state == "succeeded" and not errors
    exit_code = 0 if succeeded else primary_code if type(primary_code) is int and 1 <= primary_code <= 255 else 1
    return {"protocol": "bounded-owned-command-cli-v1", "status": "succeeded" if succeeded else "failed",
            "run_root": str(owner.root), "primary_returncode": primary_code,
            "lifecycle_complete": complete, "lease_status": "released" if owner.closed else "active",
            "manifest_state": owner.state, "exit_code": exit_code, "errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--timeout-s", required=True, type=float)
    parser.add_argument("--join-timeout-s", type=float, default=5.)
    parser.add_argument("--max-memory-bytes", required=True, type=int)
    parser.add_argument("--max-rss-bytes", type=int)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--cwd")
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--portable-primary-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.command or args.command[0] != "--":
        parser.error("an explicit command must follow --")
    try:
        result = run_owned_command(args.command[1:], output_parent=args.output_parent,
            timeout_s=args.timeout_s, join_timeout_s=args.join_timeout_s,
            max_memory_bytes=args.max_memory_bytes, max_rss_bytes=args.max_rss_bytes,
            threads=args.threads, cwd=args.cwd, inputs=args.input,
            portable_primary_only=args.portable_primary_only)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
