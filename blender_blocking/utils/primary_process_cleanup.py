"""Bounded primary-process cleanup; ordinary descendant ownership is unavailable.

For a complete tree use owned_process or owned_process_supervisor. This helper
only proves an actual Popen wait and whether inherited output pipes drained.
"""
import math
import subprocess
import time


def finish_primary_process(process, *, timeout_s=5., drain_pipes=True):
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or not 0 < timeout_s <= 30):
        raise ValueError("primary cleanup requires a finite 0..30 second allowance")
    deadline = time.monotonic() + timeout_s
    receipt = {"protocol": "bounded_primary_cleanup_v1", "primary_joined": False,
               "pipes_drained": not drain_pipes, "returncode": None, "errors": [],
               "timeout_s": timeout_s, "ordinary_tree_qualified": False,
               "scope": "primary Popen and output pipes only"}
    stdout = stderr = None
    cancellation = None
    try:
        if process.poll() is None:
            process.kill()
    except BaseException as error:
        receipt["errors"].append("termination: " + repr(error))
        if not isinstance(error, Exception):
            cancellation = error
    if drain_pipes:
        try:
            stdout, stderr = process.communicate(timeout=max(0., deadline-time.monotonic()))
            receipt["pipes_drained"] = True
        except subprocess.TimeoutExpired as error:
            stdout, stderr = error.output, error.stderr
            receipt["errors"].append("pipe drain: " + repr(error))
        except BaseException as error:
            receipt["errors"].append("pipe drain: " + repr(error))
            if not isinstance(error, Exception) and cancellation is None:
                cancellation = error
    try:
        receipt["returncode"] = process.wait(timeout=max(0., deadline-time.monotonic()))
        receipt["primary_joined"] = True
    except BaseException as error:
        receipt["errors"].append("primary join: " + repr(error))
        if not isinstance(error, Exception) and cancellation is None:
            cancellation = error
    receipt["transport_closed"] = receipt["primary_joined"] and receipt["pipes_drained"]
    if cancellation is not None:
        cancellation.primary_process_receipt = receipt
        raise cancellation
    return stdout, stderr, receipt
