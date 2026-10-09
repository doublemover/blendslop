"""Bounded opt-in ownership for one synchronous deterministic NPZ cache write."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

import numpy as np

try:
    from utils.artifact_publication import publish_file_no_clobber
    from utils.run_ownership import OwnedRun
except ImportError:  # pragma: no cover - package import path
    from ..utils.artifact_publication import publish_file_no_clobber
    from ..utils.run_ownership import OwnedRun

MAX_ARRAY_BYTES = 67108864
MAX_METADATA_BYTES = 1048576
MAX_HEADER_BYTES = 65536


class _BoundedStream:
    """Limit positions and actual writes/reads; never expose an unguarded fd."""
    def __init__(self, stream, max_bytes):
        self.stream, self.max_bytes = stream, max_bytes
        self.read_bytes = 0
        self.read_budget = max_bytes * 3 + MAX_HEADER_BYTES

    def tell(self):
        return self.stream.tell()

    def seek(self, offset, whence=0):
        position = self.stream.seek(offset, whence)
        if not 0 <= position <= self.max_bytes:
            raise ValueError("cache stage seek exceeds its byte bound")
        return position

    def write(self, data):
        if self.tell() + len(data) > self.max_bytes:
            raise ValueError("cache stage write exceeds its byte bound")
        return self.stream.write(data)

    def read(self, amount=-1):
        remaining = min(self.max_bytes - self.tell() + 1,
                        self.read_budget - self.read_bytes + 1)
        requested = remaining if amount < 0 else min(amount, remaining)
        if requested < 0:
            raise ValueError("cache stage read exceeds its byte bound")
        data = self.stream.read(requested)
        self.read_bytes += len(data)
        if self.tell() > self.max_bytes or self.read_bytes > self.read_budget:
            raise ValueError("cache stage read exceeds its byte bound")
        return data

    def flush(self):
        self.stream.flush()

    def seekable(self):
        return True



def _verify_stage(stage, array, metadata_array, stage_bound):
    if stage.stat().st_size > stage_bound:
        raise ValueError("cache stage exceeds its declared file bound")
    with stage.open("rb") as raw:
        reader = _BoundedStream(raw, stage_bound)
        with zipfile.ZipFile(reader) as archive:
            if sorted(archive.namelist()) != ["data.npy", "metadata.npy"]:
                raise ValueError("cache stage contains unexpected NPZ members")
            for name, expected in (("data.npy", array), ("metadata.npy", metadata_array)):
                info = archive.getinfo(name)
                if info.file_size > expected.nbytes + MAX_HEADER_BYTES:
                    raise ValueError("cache NPZ uncompressed member exceeds its byte bound")
                with archive.open(name) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        header = np.lib.format.read_array_header_1_0(member, max_header_size=MAX_HEADER_BYTES)
                    elif version == (2, 0):
                        header = np.lib.format.read_array_header_2_0(member, max_header_size=MAX_HEADER_BYTES)
                    else:
                        raise ValueError("unsupported bounded cache NPY header version")
                    shape, _fortran, dtype = header
                    if shape != expected.shape or dtype != expected.dtype or dtype.hasobject:
                        raise ValueError("cache NPY dimensions/dtype differ from bounded input")
        reader.seek(0)
        with np.load(reader, allow_pickle=False, max_header_size=MAX_HEADER_BYTES) as archive:
            restored = archive["data"]
            if restored.tobytes(order="C") != array.tobytes(order="C"):
                raise ValueError("cache stage does not round-trip input bytes")
            if str(archive["metadata"].item()) != str(metadata_array.item()):
                raise ValueError("cache stage does not round-trip metadata")
        return reader.read_bytes


def _receipt(owner, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf8")
    owner.reserve_bytes(len(payload))
    partial = owner.root / "cache-store.json.partial"
    with partial.open("wb") as stream:
        _BoundedStream(stream, MAX_METADATA_BYTES).write(payload)
    os.replace(partial, owner.root / "cache-store.json")
    owner.register_file("cache-store.json", "diagnostic")


def write_owned_chunk_cache(destination, data, metadata_json, *, receipt_artifacts):
    """Publish the exact key once; collisions retain prior bytes and fail explicitly."""
    array = np.asarray(data)
    encoded_metadata = metadata_json.encode("utf8")
    destination = Path(destination)
    if (not re.fullmatch(r"[0-9a-f]{64}\.npz", destination.name) or
            array.ndim != 3 or array.dtype.hasobject or array.nbytes > MAX_ARRAY_BYTES or
            len(encoded_metadata) > MAX_METADATA_BYTES):
        raise ValueError("owned cache requires a deterministic key, bounded 3D non-object array and metadata")
    # Freeze a bounded value snapshot; no source array or caller metadata is mutated.
    array = array.copy()
    metadata_array = np.array(metadata_json)
    stage_bound = array.nbytes * 2 + metadata_array.nbytes + MAX_HEADER_BYTES
    destination.parent.mkdir(parents=True, exist_ok=True)
    owner = OwnedRun(destination.parent.resolve() / ".cache-runs",
                     producer="owned_volume_chunk_cache_store",
                     max_generated_bytes=max(1048576, stage_bound + 2097152))
    receipt_artifacts["cache_store_ownership"] = owner.root / "cache-store.json"
    stage = owner.root / "chunk.npz"
    receipt = {"protocol": "owned-volume-chunk-cache-v1", "status": "writing",
               "requested_path": str(destination), "published_path": None,
               "stage": stage.name, "shape": list(array.shape), "dtype": str(array.dtype),
               "array_uncompressed_bytes": array.nbytes, "metadata_uncompressed_bytes": metadata_array.nbytes,
               "metadata_utf8_bytes": len(encoded_metadata),
               "metadata_sha256": hashlib.sha256(encoded_metadata).hexdigest(),
               "stage_byte_bound": stage_bound, "subprocesses_started": 0,
               "scope": "one synchronous cache serialization; historical/shared entries never adopted"}
    error = None
    try:
        _receipt(owner, receipt)
        owner.reserve_bytes(stage_bound)
        with stage.open("wb") as raw:
            writer = _BoundedStream(raw, stage_bound)
            np.savez_compressed(writer, data=array, metadata=metadata_array)
            writer.flush()
        receipt["verification_read_bytes"] = _verify_stage(stage, array, metadata_array, stage_bound)
        owner.register_file(stage.name, "final_output")
        publish_file_no_clobber(stage, destination)
        owner.manifest["published_output"] = {
            "path": str(destination), "stage": stage.name,
            "bytes": owner.records[stage.name]["bytes"], "sha256": owner.records[stage.name]["sha256"]}
        receipt.update(published_path=str(destination), status="published",
                       bytes=owner.records[stage.name]["bytes"], sha256=owner.records[stage.name]["sha256"])
        return destination
    except BaseException as exc:
        error = exc
        raise
    finally:
        auxiliary = []
        if stage.exists():
            try:
                owner.register_file(stage.name, "final_output" if receipt["published_path"] else "diagnostic")
            except Exception as exc:
                auxiliary.append(exc)
        failure = error or (auxiliary[0] if auxiliary else None)
        receipt.update(status="cancelled" if isinstance(failure, (KeyboardInterrupt, SystemExit))
                       else "failed" if failure else "succeeded", error=None if failure is None else repr(failure),
                       publication_collision=isinstance(error, FileExistsError))
        try:
            _receipt(owner, receipt)
        except Exception as exc:
            auxiliary.append(exc)
            partial = owner.root / "cache-store.json.partial"
            if partial.exists():
                try:
                    owner.register_file(partial.name, "diagnostic")
                except Exception as partial_error:
                    auxiliary.append(partial_error)
        owner.auxiliary_errors.extend(repr(exc) for exc in auxiliary)
        try:
            owner.close(error=error or (auxiliary[0] if auxiliary else None))
        except Exception as exc:
            auxiliary.append(exc)
        if error is not None:
            for secondary in auxiliary:
                if hasattr(error, "add_note"):
                    error.add_note("Owned cache receipt also failed: " + repr(secondary))
        elif auxiliary:
            raise auxiliary[0]
