"""Durable experiment bookkeeping.

The helpers in this module deliberately avoid importing model libraries so they
can be validated on login nodes before a GPU environment is activated.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1



def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def protocol_hash(protocol: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: str | os.PathLike[str], row: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "a", encoding="utf-8") as stream:
        stream.write(canonical_json(dict(row)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())










def validate_telemetry(row: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "qid", "method", "output_objective", "output_depth",
        "llm_calls", "selection_steps", "non_llm_bypasses",
        "nominal_prompt_tokens", "nominal_completion_tokens", "nominal_total_tokens",
        "retry_prompt_tokens", "retry_completion_tokens", "retry_total_tokens",
        "actual_prompt_tokens", "actual_completion_tokens", "actual_total_tokens",
        "retry_inclusive_llm_calls", "wall_seconds", "retry_seconds",
        "retry_excluded_wall_seconds",
        "status",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Telemetry row is missing required fields: {missing}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported telemetry schema: {row['schema_version']!r}")
    for prefix in ("nominal", "retry", "actual"):
        prompt = row[f"{prefix}_prompt_tokens"]
        completion = row[f"{prefix}_completion_tokens"]
        total = row[f"{prefix}_total_tokens"]
        if min(prompt, completion, total) < 0 or prompt + completion != total:
            raise ValueError(f"Invalid {prefix} token accounting.")
    if row["actual_prompt_tokens"] != row["nominal_prompt_tokens"] + row["retry_prompt_tokens"]:
        raise ValueError("actual_prompt_tokens != nominal + retry")
    if row["actual_completion_tokens"] != (
        row["nominal_completion_tokens"] + row["retry_completion_tokens"]
    ):
        raise ValueError("actual_completion_tokens != nominal + retry")
    if row["selection_steps"] != row["llm_calls"] + row["non_llm_bypasses"]:
        raise ValueError("selection_steps must equal llm_calls + non_llm_bypasses")
    if row["retry_inclusive_llm_calls"] < row["llm_calls"]:
        raise ValueError("retry-inclusive calls cannot be below nominal calls")
    wall = float(row["wall_seconds"])
    retry = float(row["retry_seconds"])
    retry_excluded = float(row["retry_excluded_wall_seconds"])
    if min(wall, retry, retry_excluded) < 0:
        raise ValueError("Wall/retry timing values must be non-negative.")
    if abs(retry_excluded - max(0.0, wall - retry)) > 1e-9:
        raise ValueError("retry_excluded_wall_seconds must equal max(0, wall - retry).")


def trec_lines(qid: str, docids: Sequence[str], tag: str) -> str:
    if len(set(docids)) != len(docids):
        raise ValueError(f"Duplicate docids for qid={qid!r}")
    return "".join(
        f"{qid}\tQ0\t{docid}\t{rank}\t{-rank}\t{tag}\n"
        for rank, docid in enumerate(docids, start=1)
    )


@dataclass(frozen=True)
class QueryCheckpointStore:
    root: Path
    protocol_digest: str

    @classmethod
    def open(
        cls,
        root: str | os.PathLike[str],
        protocol: Mapping[str, Any],
        *,
        resume: bool,
    ) -> "QueryCheckpointStore":
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        digest = protocol_hash(protocol)
        metadata_path = root_path / "protocol.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("protocol_hash") != digest:
                raise ValueError("Checkpoint protocol hash mismatch.")
            if not resume:
                raise FileExistsError(
                    f"Checkpoint directory {root_path} is non-empty; use --resume or a new attempt."
                )
        else:
            atomic_write_json(metadata_path, {"protocol_hash": digest, "protocol": protocol})
        (root_path / "queries").mkdir(exist_ok=True)
        return cls(root=root_path, protocol_digest=digest)

    def _safe_qid(self, qid: str) -> str:
        digest = hashlib.sha256(qid.encode("utf-8")).hexdigest()[:16]
        return f"{digest}.json"

    def path_for(self, qid: str) -> Path:
        return self.root / "queries" / self._safe_qid(qid)

    def save(self, qid: str, docids: Sequence[str], telemetry: Mapping[str, Any]) -> None:
        validate_telemetry(telemetry)
        if telemetry["qid"] != qid or telemetry["status"] != "ok":
            raise ValueError("Only matching successful telemetry can be checkpointed.")
        payload = {
            "protocol_hash": self.protocol_digest,
            "qid": qid,
            "docids": list(docids),
            "telemetry": dict(telemetry),
        }
        atomic_write_json(self.path_for(qid), payload)

    def load(self, qid: str, expected_depth: int) -> dict[str, Any] | None:
        path = self.path_for(qid)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_hash") != self.protocol_digest or payload.get("qid") != qid:
            raise ValueError(f"Invalid checkpoint for qid={qid!r}.")
        docids = payload.get("docids", [])
        if len(docids) != expected_depth or len(set(docids)) != expected_depth:
            raise ValueError(f"Checkpoint row-count/uniqueness failure for qid={qid!r}.")
        validate_telemetry(payload.get("telemetry", {}))
        return payload


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def ensure_unique_qids(rows: Iterable[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        qid = str(row["qid"])
        if qid in seen:
            raise ValueError(f"Duplicate qid in telemetry: {qid}")
        seen.add(qid)
