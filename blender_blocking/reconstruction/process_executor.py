"""Persistent isolated workers with one queue for candidates and fitting starts.

IPC contains only trusted local Python values and numeric geometry. A waiting
candidate lends its worker to fitting jobs; workers never create child pools.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import uuid


@dataclass(frozen=True)
class JobOutcome:
    status: str
    value: object = None
    error: str = ""
    elapsed_s: float = 0.0
    worker_pid: int = 0
    queue_s: float = 0.0
    total_wall_s: float = 0.0
    stop_reason: str = ""
    partial: bool = False
    recorded_evaluations: int | None = None


@dataclass(frozen=True)
class WorkerProcessBudget:
    """Explicit opt-in bounds for the entire pool, including replacements.

    Memory/RSS caps are divided across concurrent workers. The wall allowance
    starts at the first launch and is never reset by a worker restart. Each tree
    has a bounded join allowance. The supplied root becomes a scratch parent
    for fresh IPC; no cleanup or artifact lease is implied.
    """
    wall_s: float
    max_memory_bytes: int
    join_timeout_s: float = 5.
    max_rss_bytes: int | None = None
    max_restarts: int = 8

    def validate(self, workers):
        from blender_blocking.utils.owned_process_supervisor import validate_process_bounds
        validate_process_bounds([sys.executable], timeout_s=self.wall_s,
            max_memory_bytes=self.max_memory_bytes, join_timeout_s=self.join_timeout_s,
            max_rss_bytes=self.max_rss_bytes)
        if (isinstance(self.max_restarts, bool) or not isinstance(self.max_restarts, int)
                or not 0 <= self.max_restarts <= 32 or self.max_memory_bytes < workers
                or (self.max_rss_bytes is not None and self.max_rss_bytes < workers)):
            raise ValueError("worker budget requires nonempty per-worker caps and 0..32 restarts")


def caller_process_budget(*, context=None, process_budget=None):
    """Resolve an explicit pool allowance; context is a caller option, not a cap guess."""
    contextual = getattr(context, "worker_process_budget", None)
    for value in (process_budget, contextual):
        if value is not None:
            if not isinstance(value, WorkerProcessBudget):
                raise TypeError("worker process budget must be an explicit WorkerProcessBudget")
            value.validate(1)
    if process_budget is not None and contextual is not None and process_budget != contextual:
        raise ValueError("explicit and contextual worker process budgets disagree")
    return process_budget if process_budget is not None else contextual


def request_process_budget(requests, *, process_budget=None):
    """One caller allowance covers every request in a shared ensemble/routing pool."""
    effective = caller_process_budget(process_budget=process_budget)
    for request in requests:
        value = caller_process_budget(context=request.context)
        if value is not None:
            if effective is not None and effective != value:
                raise ValueError("candidate contexts require different whole-pool budgets")
            effective = value
    return effective


def executor_scope(max_workers=2, *, context=None, executor=None, process_budget=None,
                   require_submit_result=False):
    """Reuse the actual client/pool; never adopt or reset an existing allowance.

    WorkerClient remains a map-only coordinated subjob client. Candidate callers
    require submit/result and refuse recursive ensemble use before pool allocation.
    A null budget preserves the ordinary unowned constructor path.
    """
    from contextlib import nullcontext
    budget = caller_process_budget(context=context, process_budget=process_budget)
    worker = current_worker_client()
    contextual = getattr(context, "process_executor", None)
    supplied = executor if executor is not None else contextual
    if worker is not None and supplied is not None and supplied is not worker:
        raise ValueError("an active worker cannot substitute a different process executor")
    shared = worker if worker is not None else supplied
    if shared is not None:
        if budget is not None and getattr(shared, "process_budget", None) != budget:
            raise ValueError("shared executor must already own the requested whole-pool budget")
        if require_submit_result and not all(callable(getattr(shared, name, None))
                                             for name in ("submit", "result")):
            raise TypeError("candidate execution requires submit/result; WorkerClient supports scoped map subjobs only")
        return nullcontext(shared)
    if budget is None:
        return PersistentProcessExecutor(max_workers)
    budget.validate(max(1, min(4, int(max_workers))))
    return PersistentProcessExecutor(max_workers, process_budget=budget)


def write_packet(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def read_packet(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


_WORKER_CLIENT = None


def current_worker_client():
    return _WORKER_CLIENT


def progress_path():
    client = current_worker_client()
    stack = getattr(client, "stack", None)
    root = getattr(client, "root", None)
    if not stack or root is None:
        return None
    return Path(root) / "results" / (stack[-1] + ".progress.pkl")


def publish_progress(value, *, recorded_evaluations=None, path=None):
    """Persist a scored state atomically; never manufacture a result for unscored work."""
    destination = path or progress_path()
    if destination is not None:
        write_packet(destination, JobOutcome("partial", value, partial=True,
                     recorded_evaluations=recorded_evaluations, stop_reason="scored_checkpoint"))


class WorkerClient:
    def __init__(self, root, worker):
        self.root = Path(root)
        self.worker = str(worker)
        self.stack = []
        self.process_budget = None
        self._process_budget_bound = False

    def bind_process_budget(self, budget):
        """Carry the coordinator declaration, without creating a new deadline/owner."""
        if budget is not None:
            if not isinstance(budget, WorkerProcessBudget):
                raise TypeError("worker transport requires an explicit pool budget")
            budget.validate(1)
        if self._process_budget_bound and self.process_budget != budget:
            raise ValueError("a worker client cannot replace its owning pool budget")
        self.process_budget = budget
        self._process_budget_bound = True

    def execute(self, envelope):
        self.stack.append(envelope["id"])
        started = time.monotonic()
        try:
            if envelope["deadline"] is not None and time.time() >= envelope["deadline"]:
                outcome = JobOutcome("timeout", error="job deadline exhausted before execution")
            else:
                outcome = JobOutcome("success", execute_job(envelope["kind"], envelope["payload"], deadline=envelope["deadline"]))
        except Exception as exc:
            import traceback
            outcome = JobOutcome("failed", error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        finally:
            self.stack.pop()
        outcome = JobOutcome(outcome.status, outcome.value, outcome.error, time.monotonic() - started, os.getpid())
        write_packet(self.root / "results" / (envelope["id"] + ".pkl"), outcome)

    def next_job(self):
        path = self.root / "workers" / self.worker / "inbox.pkl"
        if not path.exists():
            return None
        packet = read_packet(path)
        path.unlink()
        return packet

    def map(self, jobs, *, timeout_s=None):
        group = uuid.uuid4().hex
        write_packet(self.root / "spawns" / (group + ".pkl"), {
            "group": group, "parent": self.stack[-1], "worker": self.worker,
            "jobs": jobs, "timeout_s": timeout_s,
        })
        response = self.root / "groups" / (group + ".pkl")
        while not response.exists():
            job = self.next_job()
            if job is not None:
                if job["kind"] not in {"fit_start", "program_geometry", "native_union", "dvx_fit"}:
                    raise RuntimeError("a paused candidate may execute only coordinated scoped subjobs")
                self.execute(job)
            else:
                time.sleep(0.005)
        outcomes = read_packet(response)
        response.unlink()
        return outcomes


def execute_job(kind, payload, *, deadline=None):
    if deadline is not None and kind in {"fit_start", "fit_multistart"}:
        from dataclasses import replace
        payload = dict(payload)
        config = payload["config"]
        remaining = max(0.001, deadline - time.time())
        limit = config.optimizer.max_elapsed_s
        payload["config"] = replace(config, optimizer=replace(config.optimizer,
            max_elapsed_s=remaining if limit is None else min(limit, remaining)))
    if kind == "dvx_fit":
        payload = dict(payload)
        destination = progress_path()
        paths = list(payload.get("progress_paths", ()))
        if destination is not None and str(destination) not in paths:
            paths.append(str(destination))
        payload["progress_paths"] = paths
        if deadline is not None:
            payload = dict(payload)
            remaining = max(.001, deadline-time.time())
            payload["timeout_s"] = min(payload.get("timeout_s") or remaining, remaining)
        from blender_blocking.reconstruction.differentiable.dvx_adapter import fit_job
        return fit_job(payload)
    if kind == "native_union":
        if deadline is not None:
            left, right, options = payload
            options = dict(options)
            options['qualification_timeout_s'] = min(float(options.get('qualification_timeout_s', 15.)),
                                                       max(.001, (deadline-time.time())/2.))
            payload = (left, right, options)
        from blender_blocking.reconstruction.grouped_solids import execute_union_pair
        return execute_union_pair(payload)
    if kind == "program_geometry":
        from blender_blocking.reconstruction.backends.shape_program.geometry_search import evaluate_program_job
        return evaluate_program_job(payload)
    if kind == "fit_multistart":
        from blender_blocking.placement.resfit.optimizer import fit_residual_primitives_multistart
        return fit_residual_primitives_multistart(**payload)
    if kind == "fit_start":
        from blender_blocking.placement.resfit.optimizer import fit_residual_primitives
        return fit_residual_primitives(**payload)
    if kind != "candidate":
        raise ValueError(f"unsupported process job: {kind}")
    from dataclasses import replace
    from types import SimpleNamespace
    from blender_blocking.reconstruction.registry import register_builtin_backends
    from blender_blocking.reconstruction.ensemble import _run_candidate_request
    from blender_blocking.reconstruction.native_geometry import GeometryArrays, GeometryCache
    from blender_blocking.evaluation.cost_model import CostRecorder
    request, settings, measured, backend = payload
    declaration = settings.get("worker_process_budget")
    pool_budget = None if declaration is None else WorkerProcessBudget(**declaration)
    client = current_worker_client()
    if client is not None:
        client.bind_process_budget(pool_budget)
    elif pool_budget is not None:
        pool_budget.validate(1)
    settings = {**settings, "worker_process_budget": pool_budget}
    if deadline is not None:
        remaining = max(0.001, deadline - time.time())
        limit = request.budget.timeout_s
        request = replace(request, budget=replace(request.budget,
            timeout_s=remaining if limit is None else min(limit, remaining)))
    available = False
    try:
        import bpy
        bpy.ops.wm.read_factory_settings(use_empty=True)
        available = True
    except ImportError:
        pass
    recorder = CostRecorder(track_memory=bool(settings.get("diagnostic_allocations", False)))
    context = SimpleNamespace(**settings)
    if client is not None:
        context.process_executor = client
    context.blender_available = available
    context.geometry_cache = GeometryCache()
    context.cost_recorder = recorder
    request = replace(request, context=context)
    register_builtin_backends()
    if backend is not None:
        from blender_blocking.reconstruction.registry import register_backend
        register_backend(backend, replace=True,
                         aliases=(request.backend_name,) if request.backend_name != backend.name else ())
    result = _run_candidate_request(request,
        project_diagnostics=(not measured or settings.get("projection_diagnostics", False)))
    if result.succeeded and result.mesh_path is not None:
        partial_result = replace(result, payload=None, geometry=None)
        try:
            pickle.dumps(partial_result, protocol=pickle.HIGHEST_PROTOCOL)
        except (TypeError, pickle.PicklingError):
            pass
        else:
            publish_progress(partial_result)
    if measured and result.succeeded and result.mesh_path is not None:
        from blender_blocking.reconstruction.measured_selection import render_evidence
        try:
            result = render_evidence(result, request)
            if not result.metric_result.per_view or not all(v["passed"] for v in result.metric_result.per_view.values()):
                result = replace(result, status="failed", errors=(*result.errors, "fixed-camera required-view gates failed"))
        except Exception as exc:
            result = replace(result, status="failed", errors=(*result.errors, f"fresh selection evidence failed: {exc}"))
    geometry = result.geometry
    if hasattr(geometry, "obj") and hasattr(geometry, "data"):
        data = geometry.data
        geometry.release()
        geometry = data
    if geometry is None and result.succeeded and result.mesh_path is not None:
        from blender_blocking.evaluation.comparable_geometry import read_obj
        vertices, faces = read_obj(result.mesh_path)
        geometry = GeometryArrays.capture(vertices, faces)
    extras = {**result.metric_result.extras, "process_execution": {
        "pid": os.getpid(), "scene_process_local": True, "queue": "coordinated_candidates_and_starts",
        "cost_report": recorder.report(),
    }}
    transport_payload = result.payload
    if hasattr(transport_payload, "bl_rna"):
        transport_payload = None
    else:
        try:
            pickle.dumps(transport_payload, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            transport_payload = None
    return replace(result, payload=transport_payload, geometry=geometry,
                   metric_result=replace(result.metric_result, extras=extras))


def worker_main(root, worker):
    global _WORKER_CLIENT
    client = WorkerClient(root, worker)
    _WORKER_CLIENT = client
    directory = client.root / "workers" / str(worker)
    write_packet(directory / "ready.pkl", {"pid": os.getpid()})
    try:
        while not (directory / "stop").exists():
            envelope = client.next_job()
            if envelope is None:
                time.sleep(0.005)
            else:
                client.execute(envelope)
    finally:
        # Numeric helpers belong to this worker and cannot outlive its normal
        # shutdown. Hard owner loss is also bounded by the helper heartbeat.
        import sys
        module=sys.modules.get('blender_blocking.reconstruction.differentiable.helper_session')
        if module is not None:
            module.close_helper_sessions()


class PersistentProcessExecutor:
    """Bounded persistent processes; individual failures do not break the pool."""
    def __init__(self, max_workers=2, *, blender_binary=None, root=None, threads=4,
                 process_budget=None):
        self.max_workers = max(1, min(4, int(max_workers)))
        if process_budget is not None:
            if not isinstance(process_budget, WorkerProcessBudget):
                raise TypeError("process_budget must be an explicit WorkerProcessBudget")
            process_budget.validate(self.max_workers)
        self.process_budget = process_budget
        self._budget_deadline = None
        self._budget_exhausted = False
        self._launch_count = 0
        self.process_scopes = []
        self.blender_binary = blender_binary
        self.threads = max(1, min(int(threads), 4 // self.max_workers))
        self.total_threads = self.max_workers * self.threads
        self.ownership_root = None
        if self.process_budget is not None:
            # Supplied root is a scratch parent, never transport/PID adoption.
            supplied = (Path(root) if root else Path(tempfile.gettempdir())).absolute()
            if (supplied != supplied.resolve() or supplied.is_symlink()
                    or getattr(supplied, "is_junction", lambda: False)()):
                raise ValueError("owned worker root must not be redirected")
            supplied.mkdir(parents=True, exist_ok=True)
            self.root = Path(tempfile.mkdtemp(prefix="blendslop-owned-workers-", dir=supplied))
            self.ownership_root = self.root / "process-ownership"
            self.ownership_root.mkdir()
        else:
            self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="blendslop-workers-"))
        for name in ("workers", "results", "spawns", "groups", "artifacts"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.pending = deque()
        self.jobs = {}
        self.workers = {}
        self.groups = {}
        self.closed = False
        self._sequence = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is None:
                raise
            if hasattr(error, "add_note"):
                error.add_note("Worker ownership close also failed: " + repr(cleanup_error))
        return False

    def start(self):
        if self.closed:
            raise RuntimeError("executor is closed")
        if self.workers:
            return
        if self.blender_binary is None:
            try:
                import bpy
                self.blender_binary = bpy.app.binary_path
            except ImportError:
                pass
        if self.process_budget is not None and self._budget_deadline is None:
            self._budget_deadline = time.monotonic() + self.process_budget.wall_s
        try:
            for index in range(self.max_workers):
                self._start_worker(str(index))
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note("Worker startup cleanup also failed: " + repr(cleanup_error))
            raise

    def _start_worker(self, worker):
        if self.process_budget is not None:
            if (self._budget_deadline is None or time.monotonic() >= self._budget_deadline
                    or self._launch_count >= self.max_workers + self.process_budget.max_restarts):
                return False
        directory = self.root / "workers" / worker
        directory.mkdir(exist_ok=True)
        for name in ("ready.pkl", "inbox.pkl", "stop"):
            (directory / name).unlink(missing_ok=True)
        script = Path(__file__).resolve().parents[2] / "scripts" / "reconstruction_worker.py"
        if self.blender_binary:
            command = [str(self.blender_binary), "--background", "--factory-startup", "--threads", str(self.threads),
                       "--python", str(script), "--", "--root", str(self.root), "--worker", worker]
        else:
            command = [sys.executable, "-B", str(script), "--root", str(self.root), "--worker", worker]
        environment = dict(os.environ)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[name] = str(self.threads)
        scope = None
        if self.process_budget is not None:
            from blender_blocking.utils.owned_process import OwnedProcess
            generation = self.ownership_root / ("worker-" + worker + "-" + uuid.uuid4().hex)
            generation.mkdir()
            scope = OwnedProcess(command, log_path=generation / "worker.log",
                timeout_s=max(.001, self._budget_deadline-time.monotonic()),
                max_memory_bytes=self.process_budget.max_memory_bytes // self.max_workers,
                join_timeout_s=self.process_budget.join_timeout_s,
                max_rss_bytes=(None if self.process_budget.max_rss_bytes is None else
                               self.process_budget.max_rss_bytes // self.max_workers), env=environment)
            self.process_scopes.append(scope)
            self._launch_count += 1
            self._publish_process_receipt(scope)
            try:
                process = scope.start()
            except BaseException as error:
                try:
                    self._publish_process_receipt(scope)
                except BaseException as publication_error:
                    if hasattr(error, "add_note"):
                        error.add_note("Worker launch receipt also failed: " + repr(publication_error))
                raise
            self._publish_process_receipt(scope)
            log = None
        else:
            log = (directory / "worker.log").open("ab")
            try:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except BaseException:
                log.close()
                raise
        self.workers[worker] = {"process": process, "log": log, "scope": scope,
                                "stack": [], "waiting": set(), "started": time.monotonic(),
                                "ready": False, "ownership_stopping": False}
        return True

    @staticmethod
    def _publish_process_receipt(scope):
        destination = scope.log_path.parent / "lifecycle.json"
        temporary = destination.with_suffix(".json." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(scope.snapshot(), indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)

    def ownership_receipt(self):
        scopes = [scope.snapshot() for scope in self.process_scopes]
        return {"protocol": "persistent_worker_ownership_v1", "opted_in": self.process_budget is not None,
                "closed": self.closed, "lifecycle_complete": bool(self.process_budget is not None
                    and self.closed and all(row["lifecycle_complete"] for row in scopes)),
                "artifact_lease_release": "unsupported", "files_retained": True,
                "pool_wall_s": None if self.process_budget is None else self.process_budget.wall_s,
                "pool_memory_cap_bytes": None if self.process_budget is None else self.process_budget.max_memory_bytes,
                "maximum_close_join_s": None if self.process_budget is None else
                    self.max_workers * self.process_budget.join_timeout_s,
                "launch_count": self._launch_count, "processes": scopes}

    def submit(self, kind, payload, *, timeout_s=None, deadline=None):
        if self.closed:
            raise RuntimeError("executor is closed")
        self.start()
        job_id = uuid.uuid4().hex
        # Total deadlines include queue time; per-job time starts at dispatch.
        limits = [v for v in (deadline,) if v is not None]
        self.jobs[job_id] = {"id": job_id, "kind": kind, "payload": payload,
            "deadline": min(limits) if limits else None, "outcome": None, "worker": None,
            "timeout_s": timeout_s, "submitted_at": time.time(), "dispatched_at": None}
        try:
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            self.jobs[job_id]["outcome"] = JobOutcome("failed", error=f"job payload is not process-safe: {exc}")
        else:
            self.pending.append(job_id)
        return job_id

    def map(self, jobs, *, timeout_s=None):
        deadline = None if timeout_s is None else time.time() + max(0.0, float(timeout_s))
        ids = [self.submit(kind, payload, timeout_s=limit, deadline=deadline) for kind, payload, limit in jobs]
        return [self.result(job_id) for job_id in ids]

    def result(self, job_id):
        while self.jobs[job_id]["outcome"] is None:
            self.poll()
            time.sleep(0.005)
        return self.jobs[job_id]["outcome"]

    def done(self, job_id):
        self.poll()
        return self.jobs[job_id]["outcome"] is not None

    def _finish(self, job_id, outcome):
        job = self.jobs[job_id]
        if job["outcome"] is not None:
            return
        from dataclasses import replace
        finished = time.time()
        checkpoint = self.root / "results" / (job_id + ".progress.pkl")
        if outcome.status in {"timeout", "failed"} and checkpoint.exists():
            saved = read_packet(checkpoint)
            if isinstance(saved, dict):
                saved = JobOutcome(saved['status'], saved.get('value'), partial=True,
                                   recorded_evaluations=saved.get('recorded_evaluations'))
            if saved.status == "partial" and saved.value is not None:
                outcome = replace(outcome, value=saved.value, partial=True,
                                  recorded_evaluations=saved.recorded_evaluations)
        checkpoint.unlink(missing_ok=True)
        dispatched = job.get("dispatched_at")
        submitted = job.get("submitted_at", finished)
        job["outcome"] = replace(outcome,
            queue_s=max(0., (finished if dispatched is None else dispatched) - submitted),
            total_wall_s=max(0., finished - submitted),
            elapsed_s=outcome.elapsed_s or (0. if dispatched is None else max(0., finished - dispatched)),
            stop_reason=outcome.stop_reason or outcome.error or "completed")
        worker = self.workers.get(job["worker"])
        if worker is not None and job_id in worker["stack"]:
            worker["stack"].remove(job_id)

    def _stop_worker(self, worker):
        state = self.workers[worker]
        process = state["process"]
        if state.get("scope") is not None:
            state["ownership_stopping"] = True
            result = state["scope"].stop(reason="executor_worker_stopped")
            self._publish_process_receipt(state["scope"])
            return result["lifecycle_complete"]
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=10)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        state["log"].close()
        return True

    def poll(self):
        now = time.time()
        if self.process_budget is not None and self._budget_deadline is not None:
            self._budget_exhausted = time.monotonic() >= self._budget_deadline
        for path in list((self.root / "results").glob("*.pkl")):
            if path.name.endswith(".progress.pkl"):
                continue
            job_id = path.stem
            outcome = read_packet(path)
            path.unlink()
            if job_id in self.jobs:
                if self._budget_exhausted and self.jobs[job_id]["outcome"] is None:
                    outcome = JobOutcome("timeout", error="executor lifetime exceeded before result collection")
                self._finish(job_id, outcome)
        for path in list((self.root / "spawns").glob("*.pkl")):
            packet = read_packet(path)
            path.unlink()
            parent = self.jobs.get(packet["parent"])
            if parent is None or parent["outcome"] is not None:
                continue
            state = self.workers[packet["worker"]]
            state["waiting"].add(packet["parent"])
            deadline = None if parent["deadline"] is None else parent["deadline"] - 0.1
            if packet["timeout_s"] is not None:
                limit = now + max(0.0, float(packet["timeout_s"]))
                deadline = min(deadline, limit) if deadline is not None else limit
            ids = [self.submit(kind, payload, timeout_s=limit, deadline=deadline)
                   for kind, payload, limit in packet["jobs"]]
            self.groups[packet["group"]] = (ids, packet["parent"], packet["worker"])
        for worker, state in list(self.workers.items()):
            directory = self.root / "workers" / worker
            if (directory / "ready.pkl").exists():
                state["ready"] = True
            ownership_failure = ""
            scope = state.get("scope")
            if scope is not None and not state["ownership_stopping"]:
                try:
                    observation = scope.poll()
                    ownership_failure = observation["limit_reason"] or ""
                except Exception as error:
                    scope.receipt["errors"].append("worker observation: " + repr(error))
                    ownership_failure = "owned process observation failed"
            exited = state["process"].poll() is not None
            startup_failed = not state["ready"] and time.monotonic() - state["started"] > 60
            expired = [j for j in state["stack"] if self.jobs[j]["deadline"] is not None and now >= self.jobs[j]["deadline"]]
            if (exited or startup_failed or expired or ownership_failure
                    or self._budget_exhausted or state["ownership_stopping"]):
                for job_id in list(state["stack"]):
                    self._finish(job_id, JobOutcome("timeout" if expired or self._budget_exhausted or ownership_failure in {"wall_timeout", "tree_rss_limit", "log_byte_limit"} else "failed",
                        error=("executor lifetime exceeded" if self._budget_exhausted else ownership_failure or
                               ("worker deadline exceeded" if expired else "worker exited; see " +
                                str(scope.log_path if scope is not None else directory / "worker.log")))))
                if not self._stop_worker(worker):
                    continue
                del self.workers[worker]
                if not self.closed and not startup_failed and not self._budget_exhausted and state["ready"]:
                    self._start_worker(worker)
        for job_id in list(self.pending):
            job = self.jobs[job_id]
            if self._budget_exhausted or (job["deadline"] is not None and now >= job["deadline"]):
                self.pending.remove(job_id)
                self._finish(job_id, JobOutcome("timeout", error="executor lifetime exhausted in queue" if self._budget_exhausted else "job deadline exhausted in queue"))
        for group, (ids, parent, worker) in list(self.groups.items()):
            if all(self.jobs[j]["outcome"] is not None for j in ids):
                write_packet(self.root / "groups" / (group + ".pkl"), [self.jobs[j]["outcome"] for j in ids])
                if worker in self.workers:
                    self.workers[worker]["waiting"].discard(parent)
                del self.groups[group]
        for worker, state in self.workers.items():
            if not state["ready"] or self._budget_exhausted or state["ownership_stopping"]:
                continue
            inbox = self.root / "workers" / worker / "inbox.pkl"
            if inbox.exists():
                continue
            paused = bool(state["stack"]) and all(j in state["waiting"] for j in state["stack"])
            if state["stack"] and not paused:
                continue
            job_id = next((j for j in self.pending if not paused or self.jobs[j]["kind"] in {"fit_start", "program_geometry", "native_union", "dvx_fit"}), None)
            if job_id is None:
                continue
            self.pending.remove(job_id)
            job = self.jobs[job_id]
            job["worker"] = worker
            job["dispatched_at"] = time.time()
            if job["timeout_s"] is not None:
                limit = time.time() + max(0.0, float(job["timeout_s"]))
                job["deadline"] = limit if job["deadline"] is None else min(job["deadline"], limit)
            state["stack"].append(job_id)
            write_packet(inbox, {k: job[k] for k in ("id", "kind", "payload", "deadline")})
        if not self.workers:
            for job_id in list(self.pending):
                self._finish(job_id, JobOutcome("failed", error="no worker could start"))
            self.pending.clear()

    def close(self):
        self.closed = True
        for worker in list(self.workers):
            (self.root / "workers" / worker / "stop").touch()
        errors = []
        for worker in list(self.workers):
            try:
                if self._stop_worker(worker):
                    del self.workers[worker]
                else:
                    errors.append("worker " + worker + " has an incomplete owned tree join")
            except Exception as error:
                errors.append(repr(error))
        # Startup may have failed before its process entered the worker table.
        for scope in self.process_scopes:
            if not scope.released and not any(state.get("scope") is scope for state in self.workers.values()):
                try:
                    scope.stop(reason="executor_startup_cleanup")
                    self._publish_process_receipt(scope)
                    if not scope.released:
                        errors.append("startup process tree join remains incomplete")
                except Exception as error:
                    errors.append(repr(error))
        for job_id, job in self.jobs.items():
            if job["outcome"] is None:
                self._finish(job_id, JobOutcome("cancelled", error="executor closed"))
        if errors:
            raise RuntimeError("; ".join(errors))


def candidate_payload(request, executor, *, measured=False):
    """Drop workflow/bpy handles; only target data and simple settings cross IPC."""
    from dataclasses import replace
    settings = {name: getattr(request.context, name, default) for name, default in (
        ("native_resident", True), ("projection_diagnostics", False),
        ("diagnostic_allocations", False), ("requested_mode", "ensemble"),
    )}
    from dataclasses import asdict
    owning_budget = getattr(executor, "process_budget", None)
    requested_budget = caller_process_budget(context=request.context)
    if requested_budget is not None and owning_budget != requested_budget:
        raise ValueError("candidate transport cannot adopt or change its owning pool budget")
    settings["worker_process_budget"] = None if owning_budget is None else asdict(owning_budget)
    root = request.artifact_root or executor.root / "artifacts"
    from .registry import get_backend
    try:
        backend = get_backend(request.backend_name)
    except KeyError:
        backend = None
    if backend is not None and type(backend).__module__.startswith(("blender_blocking.reconstruction.backends.", "reconstruction.backends.")):
        backend = None
    return replace(request, context=None, artifact_root=root), settings, measured, backend


def candidate_outcome(request, outcome):
    from .types import CandidateResult, CandidateMetrics
    if outcome.status == "success":
        from dataclasses import replace
        result = outcome.value
        execution = {"queue_s": outcome.queue_s, "worker_execute_wall_s": outcome.elapsed_s,
                     "total_wall_s": outcome.total_wall_s, "stop_reason": outcome.stop_reason,
                     "wall_includes_child_wait": True}
        return replace(result, metric_result=replace(result.metric_result,
            extras={**result.metric_result.extras, "queue_accounting": execution}))
    partial = outcome.value if outcome.partial else None
    partial_artifacts = ({"mesh_path": str(partial.mesh_path) if partial.mesh_path else None,
                          "primitive_path": str(partial.primitive_path) if partial.primitive_path else None,
                          "artifacts": {k: str(v) for k, v in partial.artifacts.items()},
                          "fresh_render_validation": "not_established"}
                         if hasattr(partial, "artifacts") else None)
    return CandidateResult(request.candidate_id, request.backend_name,
        "skipped" if "in queue" in outcome.error else "failed",
        errors=() if "in queue" in outcome.error else (outcome.error,),
        warnings=(outcome.error,) if "in queue" in outcome.error else (),
        metric_result=CandidateMetrics(elapsed_s=outcome.elapsed_s, extras={"queue_accounting": {
            "queue_s": outcome.queue_s, "worker_execute_wall_s": outcome.elapsed_s,
            "total_wall_s": outcome.total_wall_s, "stop_reason": outcome.stop_reason,
            "partial_scored_state_available": outcome.partial},
            "partial_candidate_artifacts": partial_artifacts}))
