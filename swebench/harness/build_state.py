from __future__ import annotations

import json
import threading
import time

from datetime import datetime, timezone
from pathlib import Path


class BuildState:
    """
    Thread-safe persistent state tracker for Docker image builds.

    Writes to disk after every status change so state survives crashes.
    Entries left in 'building' status (from a crashed run) are treated
    as retryable on the next invocation.
    """

    def __init__(self, state_file: Path, dataset_name: str, arch: str):
        self._path = Path(state_file)
        self._lock = threading.Lock()

        if self._path.exists():
            with open(self._path) as f:
                self._data = json.load(f)
        else:
            self._data = {
                "metadata": {
                    "dataset": dataset_name,
                    "arch": arch,
                    "started_at": _now_iso(),
                    "last_updated": _now_iso(),
                },
                "instances": {},
            }
            self._save_unlocked()

    def initialize(self, instance_ids: list[str]):
        """Add instance IDs as 'pending' if not already tracked."""
        with self._lock:
            for iid in instance_ids:
                if iid not in self._data["instances"]:
                    self._data["instances"][iid] = _blank_entry()
            self._save_unlocked()

    def mark_building(self, instance_id: str, image_key: str, env_image_key: str):
        with self._lock:
            entry = self._data["instances"].setdefault(instance_id, _blank_entry())
            entry["status"] = "building"
            entry["image_key"] = image_key
            entry["env_image_key"] = env_image_key
            entry["error"] = None
            self._save_unlocked()

    def mark_success(self, instance_id: str, duration: float):
        with self._lock:
            entry = self._data["instances"][instance_id]
            entry["status"] = "success"
            entry["built_at"] = _now_iso()
            entry["duration_seconds"] = round(duration, 2)
            entry["error"] = None
            self._save_unlocked()

    def mark_failed(self, instance_id: str, error: str, duration: float):
        with self._lock:
            entry = self._data["instances"][instance_id]
            entry["status"] = "failed"
            entry["built_at"] = _now_iso()
            entry["duration_seconds"] = round(duration, 2)
            entry["error"] = str(error)
            self._save_unlocked()

    def mark_env_failed(self, instance_id: str, error: str):
        """Mark an instance as failed due to its env image failing to build."""
        with self._lock:
            entry = self._data["instances"].setdefault(instance_id, _blank_entry())
            entry["status"] = "failed"
            entry["built_at"] = _now_iso()
            entry["error"] = f"env image build failed: {error}"
            self._save_unlocked()

    def mark_verified(self, instance_id: str):
        with self._lock:
            entry = self._data["instances"][instance_id]
            entry["verified"] = True
            entry["verify_error"] = None
            self._save_unlocked()

    def mark_verify_failed(self, instance_id: str, error: str):
        with self._lock:
            entry = self._data["instances"][instance_id]
            entry["status"] = "verify_failed"
            entry["verified"] = False
            entry["verify_error"] = str(error)
            self._save_unlocked()

    def mark_pushed(self, instance_id: str, registry_image: str):
        with self._lock:
            entry = self._data["instances"][instance_id]
            entry["pushed"] = True
            entry["registry_image"] = registry_image
            entry["pushed_at"] = _now_iso()
            self._save_unlocked()

    def mark_push_failed(self, instance_id: str, error: str):
        with self._lock:
            entry = self._data["instances"][instance_id]
            entry["pushed"] = False
            entry["push_error"] = str(error)
            self._save_unlocked()

    def get_unpushed(self) -> list[str]:
        """Return instance IDs that built successfully but haven't been pushed."""
        with self._lock:
            return [
                iid
                for iid, e in self._data["instances"].items()
                if e["status"] == "success" and not e.get("pushed", False)
            ]

    def get_pending(self) -> list[str]:
        """Return instance IDs that are pending or were interrupted mid-build."""
        with self._lock:
            return [
                iid
                for iid, e in self._data["instances"].items()
                if e["status"] in ("pending", "building")
            ]

    def get_failed(self) -> list[str]:
        with self._lock:
            return [
                iid
                for iid, e in self._data["instances"].items()
                if e["status"] in ("failed", "verify_failed")
            ]

    def get_successful(self) -> list[str]:
        with self._lock:
            return [
                iid
                for iid, e in self._data["instances"].items()
                if e["status"] == "success"
            ]

    def reset_status(self, instance_ids: list[str]):
        """Reset given instance IDs back to pending (for retries)."""
        with self._lock:
            for iid in instance_ids:
                if iid in self._data["instances"]:
                    self._data["instances"][iid] = _blank_entry()
            self._save_unlocked()

    def summary(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for e in self._data["instances"].values():
                s = e["status"]
                counts[s] = counts.get(s, 0) + 1
            return counts

    def _save_unlocked(self):
        """Write state to disk. Must be called while holding self._lock."""
        self._data["metadata"]["last_updated"] = _now_iso()
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        tmp.replace(self._path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _blank_entry() -> dict:
    return {
        "status": "pending",
        "image_key": None,
        "env_image_key": None,
        "built_at": None,
        "duration_seconds": None,
        "error": None,
    }
