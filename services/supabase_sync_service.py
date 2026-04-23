import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib import error, parse, request

from rag.collection_utils import build_collection_name
from database.document_db import (
    Document,
    DocumentChunk,
    SessionLocal,
    count_unresolved_document_sync_errors,
    list_active_collection_names,
    log_document_sync_error,
    mark_document_sync_error_resolved,
    utcnow,
)
from services.document_ingest_service import delete_vectors_for_object_path, process_document_ingest

logger = logging.getLogger(__name__)

_SCHEDULER_OBJECT_PATH = "__supabase_sync_scheduler__"
_SCHEDULER_OPERATION = "scan_snapshot"


class SupabaseStorageSyncService:
    def __init__(
        self,
        supabase_url: str,
        service_role_key: str,
        bucket: str,
        snapshot_file: Optional[str] = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.supabase_url = (supabase_url or "").rstrip("/")
        self.service_role_key = (service_role_key or "").strip()
        self.bucket = (bucket or "").strip()
        self.snapshot_file = snapshot_file
        self.timeout_seconds = timeout_seconds

        if not self.supabase_url or not self.service_role_key or not self.bucket:
            raise ValueError("Supabase sync config is incomplete.")

        self._snapshot_loaded = False
        self._snapshot: Dict[str, Dict[str, Any]] = {}

    def list_root_folders(self) -> List[str]:
        items = self._list_objects_by_prefix(prefix="", limit=1000)
        folders = set()

        for item in items:
            name = str(item.get("name") or "").strip()
            if not name or name == ".keep":
                continue

            if self._is_folder_list_item(item):
                folders.add(name)
                continue

            if "/" in name:
                folders.add(name.split("/", 1)[0].strip())

        return sorted(folder for folder in folders if folder)

    def list_objects(self, folder_key: str) -> List[Dict[str, Any]]:
        normalized_folder = (folder_key or "").strip().strip("/")
        if not normalized_folder:
            return []

        items = self._list_objects_by_prefix(prefix=normalized_folder, limit=1000)
        files: List[Dict[str, Any]] = []

        for item in items:
            if self._is_folder_list_item(item):
                continue

            name = str(item.get("name") or "").strip()
            if not name or name == ".keep":
                continue

            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            object_path = f"{normalized_folder}/{name.lstrip('/')}"

            files.append(
                {
                    "folder_key": normalized_folder,
                    "name": name,
                    "object_path": object_path,
                    "id": str(item.get("id") or ""),
                    "created_at": str(item.get("created_at") or ""),
                    "updated_at": str(item.get("updated_at") or item.get("created_at") or ""),
                    "size": int(metadata.get("size") or 0),
                    "content_type": str(
                        metadata.get("mimetype")
                        or metadata.get("contentType")
                        or metadata.get("content_type")
                        or ""
                    ),
                    "etag": str(metadata.get("eTag") or metadata.get("etag") or ""),
                }
            )

        files.sort(key=lambda row: str(row.get("object_path") or ""))
        return files

    def list_all_objects(self) -> List[Dict[str, Any]]:
        objects_by_path: Dict[str, Dict[str, Any]] = {}

        for folder_key in self.list_root_folders():
            for row in self.list_objects(folder_key):
                object_path = str(row.get("object_path") or "").strip()
                if object_path:
                    objects_by_path[object_path] = row

        return [objects_by_path[path] for path in sorted(objects_by_path.keys())]

    def download_object(self, object_path: str, destination_path: Optional[str] = None) -> str:
        normalized_path = (object_path or "").strip().lstrip("/")
        if not normalized_path:
            raise ValueError("object_path is required.")

        encoded_path = self._encode_object_path(normalized_path)
        binary_data = self._request_bytes("GET", f"/storage/v1/object/{self.bucket}/{encoded_path}")

        safe_name = os.path.basename(normalized_path) or "document.bin"

        if destination_path is None:
            fd, destination_path = tempfile.mkstemp(prefix="supabase_", suffix=f"_{safe_name}")
            os.close(fd)
        elif os.path.isdir(destination_path):
            destination_path = os.path.join(destination_path, safe_name)

        target_dir = os.path.dirname(destination_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        with open(destination_path, "wb") as output_file:
            output_file.write(binary_data)

        return destination_path

    def scan_and_diff_snapshot(self) -> Dict[str, Any]:
        self._load_snapshot_once()

        current_objects = self.list_all_objects()
        current_by_path = {str(row["object_path"]): row for row in current_objects}

        new_snapshot = {
            object_path: self._build_snapshot_entry(row)
            for object_path, row in current_by_path.items()
        }

        old_paths = set(self._snapshot.keys())
        new_paths = set(new_snapshot.keys())

        added_paths = sorted(new_paths - old_paths)
        deleted_paths = sorted(old_paths - new_paths)

        updated_paths = sorted(
            path
            for path in (old_paths & new_paths)
            if self._has_snapshot_changed(self._snapshot[path], new_snapshot[path])
        )

        added = [current_by_path[path] for path in added_paths]
        updated = [current_by_path[path] for path in updated_paths]
        deleted = [
            {
                "object_path": path,
                "folder_key": self._snapshot[path].get("folder_key", ""),
            }
            for path in deleted_paths
        ]

        self._snapshot = new_snapshot
        self._save_snapshot()

        return {
            "total_folders": len(self.list_root_folders()),
            "total_objects": len(current_objects),
            "added": added,
            "updated": updated,
            "deleted": deleted,
        }

    def _build_snapshot_entry(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "folder_key": str(row.get("folder_key") or ""),
            "id": str(row.get("id") or ""),
            "updated_at": str(row.get("updated_at") or ""),
            "size": int(row.get("size") or 0),
            "etag": str(row.get("etag") or ""),
        }

    @staticmethod
    def _has_snapshot_changed(old_entry: Dict[str, Any], new_entry: Dict[str, Any]) -> bool:
        tracked_fields = ("id", "updated_at", "size", "etag")
        return any(old_entry.get(field) != new_entry.get(field) for field in tracked_fields)

    @staticmethod
    def _is_folder_list_item(item: Dict[str, Any]) -> bool:
        has_missing_id = "id" in item and (item.get("id") is None or str(item.get("id")) == "")
        has_missing_metadata = item.get("metadata") is None
        return bool(has_missing_id or has_missing_metadata)

    def _list_objects_by_prefix(self, prefix: str, limit: int = 1000) -> List[Dict[str, Any]]:
        all_items: List[Dict[str, Any]] = []
        offset = 0

        while True:
            payload = {
                "prefix": prefix,
                "limit": limit,
                "offset": offset,
                "sortBy": {
                    "column": "name",
                    "order": "asc",
                },
            }
            response = self._request_json("POST", f"/storage/v1/object/list/{self.bucket}", payload)
            items = response if isinstance(response, list) else []

            typed_items = [item for item in items if isinstance(item, dict)]
            all_items.extend(typed_items)

            if len(items) < limit:
                break

            offset += limit

        return all_items

    def _load_snapshot_once(self) -> None:
        if self._snapshot_loaded:
            return

        self._snapshot_loaded = True
        self._snapshot = {}

        if not self.snapshot_file or not os.path.exists(self.snapshot_file):
            return

        try:
            with open(self.snapshot_file, "r", encoding="utf-8") as input_file:
                payload = json.load(input_file)

            if not isinstance(payload, dict):
                return

            normalized: Dict[str, Dict[str, Any]] = {}
            for object_path, value in payload.items():
                if not isinstance(value, dict):
                    continue
                path = str(object_path or "").strip()
                if not path:
                    continue
                normalized[path] = {
                    "folder_key": str(value.get("folder_key") or ""),
                    "id": str(value.get("id") or ""),
                    "updated_at": str(value.get("updated_at") or ""),
                    "size": int(value.get("size") or 0),
                    "etag": str(value.get("etag") or ""),
                }

            self._snapshot = normalized
        except Exception:
            logger.exception("Failed to load Supabase snapshot file.")
            self._snapshot = {}

    def _save_snapshot(self) -> None:
        if not self.snapshot_file:
            return

        try:
            directory = os.path.dirname(self.snapshot_file)
            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(self.snapshot_file, "w", encoding="utf-8") as output_file:
                json.dump(self._snapshot, output_file, ensure_ascii=False, indent=2)
        except Exception:
            logger.exception("Failed to persist Supabase snapshot file.")

    def _request_json(self, method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        raw_bytes = self._request_bytes(method, endpoint, payload)
        if not raw_bytes:
            return None

        try:
            return json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as error_detail:
            raise RuntimeError(f"Supabase returned invalid JSON: {error_detail}") from error_detail

    def _request_bytes(self, method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> bytes:
        target_url = f"{self.supabase_url}{endpoint}"
        body = None

        headers = {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
        }

        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(target_url, data=body, headers=headers, method=method.upper())

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return response.read()
        except error.HTTPError as http_error:
            error_body = http_error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Supabase API error {http_error.code} at {endpoint}: {error_body}"
            ) from http_error
        except error.URLError as url_error:
            raise RuntimeError(f"Supabase connection error at {endpoint}: {url_error.reason}") from url_error
        except TimeoutError as timeout_error:
            raise RuntimeError(f"Supabase request timed out at {endpoint} (>{self.timeout_seconds}s): {timeout_error}") from timeout_error

    @staticmethod
    def _encode_object_path(object_path: str) -> str:
        segments = [segment for segment in object_path.split("/") if segment]
        return "/".join(parse.quote(segment, safe="") for segment in segments)


def _parse_iso_datetime(value: Optional[str]):
    raw = (value or "").strip()
    if not raw:
        return None

    normalized = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _datetime_to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


class SupabaseSyncCoordinator:
    def __init__(self, sync_service: SupabaseStorageSyncService, poll_interval_seconds: int = 120) -> None:
        self.sync_service = sync_service
        self.poll_interval_seconds = max(60, min(180, int(poll_interval_seconds or 120)))
        self._lock = asyncio.Lock()
        self._pending_event = False
        self._queued_events = 0
        self._event_task: Optional[asyncio.Task] = None

        self._last_sync_at: Optional[datetime] = None
        self._last_error_at: Optional[datetime] = None
        self._last_error_message: Optional[str] = None
        self._last_trigger: Optional[str] = None
        self._last_result: Dict[str, Any] = {
            "added": 0,
            "updated": 0,
            "deleted": 0,
            "failed": 0,
            "total_objects": 0,
            "total_folders": 0,
        }
        self._total_runs = 0
        self._consecutive_failures = 0

    async def run_polling_loop(self, stop_event: asyncio.Event) -> None:
        logger.info("Supabase sync coordinator polling loop started. interval_seconds=%s", self.poll_interval_seconds)

        while not stop_event.is_set():
            await self.run_sync(trigger="polling", queue_if_locked=False)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.poll_interval_seconds)
            except asyncio.TimeoutError:
                continue

        logger.info("Supabase sync coordinator polling loop stopped.")

    async def request_event_sync(self, event_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        trigger = f"event:{(event_name or 'notify').strip() or 'notify'}"

        if self._lock.locked() or (self._event_task is not None and not self._event_task.done()):
            self._pending_event = True
            self._queued_events += 1
            result = {"status": "queued"}
        else:
            self._event_task = asyncio.create_task(
                self.run_sync(
                    trigger=trigger,
                    payload=payload or {},
                    queue_if_locked=True,
                )
            )
            result = {"status": "started"}

        return {
            "status": result.get("status", "queued"),
            "trigger": trigger,
            "queued_events": self._queued_events,
        }

    async def run_sync(
        self,
        trigger: str,
        payload: Optional[Dict[str, Any]] = None,
        queue_if_locked: bool = False,
    ) -> Dict[str, Any]:
        if self._lock.locked():
            if queue_if_locked:
                self._pending_event = True
                self._queued_events += 1
                return {"status": "queued"}
            return {"status": "busy"}

        try:
            async with self._lock:
                current_trigger = trigger
                current_payload = payload or {}

                while True:
                    self._pending_event = False
                    cycle_result = await asyncio.to_thread(self._execute_sync_cycle, current_trigger, current_payload)

                    self._last_sync_at = utcnow()
                    self._last_trigger = current_trigger
                    self._last_result = cycle_result
                    self._last_error_message = None
                    self._last_error_at = None
                    self._total_runs += 1
                    self._consecutive_failures = 0

                    try:
                        await asyncio.to_thread(_resolve_scheduler_sync_error)
                    except Exception:
                        logger.exception("Failed to resolve scheduler sync error state.")

                    if self._pending_event:
                        current_trigger = "event:batched"
                        current_payload = {}
                        continue

                    self._queued_events = 0

                    return {
                        "status": "completed",
                        "result": cycle_result,
                    }
        except Exception as error:
            logger.exception("Supabase sync coordinator cycle failed")
            self._last_error_message = str(error)
            self._last_error_at = utcnow()
            self._consecutive_failures += 1

            try:
                await asyncio.to_thread(_persist_scheduler_sync_error, str(error))
            except Exception:
                logger.exception("Failed to persist scheduler sync error state.")

            return {
                "status": "failed",
                "error": str(error),
            }

    def get_health_snapshot(self) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            unresolved_errors = count_unresolved_document_sync_errors(db)
            active_collections = list_active_collection_names(db, limit=20)
        finally:
            db.close()

        return {
            "running": self._lock.locked(),
            "poll_interval_seconds": self.poll_interval_seconds,
            "queued_events": self._queued_events,
            "last_sync_at": _datetime_to_iso(self._last_sync_at),
            "last_trigger": self._last_trigger,
            "last_result": self._last_result,
            "last_error_message": self._last_error_message,
            "last_error_at": _datetime_to_iso(self._last_error_at),
            "total_runs": self._total_runs,
            "consecutive_failures": self._consecutive_failures,
            "unresolved_sync_errors": unresolved_errors,
            "active_collections": active_collections,
        }

    def _execute_sync_cycle(self, trigger: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        del payload
        max_retries = 3
        retry_backoff_seconds = 2
        
        last_error = None
        for attempt in range(max_retries):
            try:
                scan_result = self.sync_service.scan_and_diff_snapshot()
                apply_result = self._apply_incremental_changes(scan_result)

                return {
                    "trigger": trigger,
                    "total_folders": int(scan_result.get("total_folders", 0)),
                    "total_objects": int(scan_result.get("total_objects", 0)),
                    "added": int(apply_result.get("added", 0)),
                    "updated": int(apply_result.get("updated", 0)),
                    "deleted": int(apply_result.get("deleted", 0)),
                    "failed": int(apply_result.get("failed", 0)),
                }
            except RuntimeError as error:
                last_error = error
                if "timed out" in str(error).lower() and attempt < max_retries - 1:
                    wait_time = retry_backoff_seconds * (2 ** attempt)
                    logger.warning(
                        f"Supabase sync timeout on attempt {attempt + 1}/{max_retries}. "
                        f"Retrying in {wait_time}s... Error: {error}"
                    )
                    time.sleep(wait_time)
                    continue
                raise
        
        if last_error:
            raise last_error

    def _apply_incremental_changes(self, scan_result: Dict[str, Any]) -> Dict[str, int]:
        added_rows = [row for row in (scan_result.get("added") or []) if isinstance(row, dict)]
        updated_rows = [row for row in (scan_result.get("updated") or []) if isinstance(row, dict)]
        deleted_rows = [row for row in (scan_result.get("deleted") or []) if isinstance(row, dict)]

        stats = {
            "added": 0,
            "updated": 0,
            "deleted": 0,
            "failed": 0,
        }

        for row in added_rows:
            success = self._upsert_and_ingest_object(row, operation="upsert")
            if success:
                stats["added"] += 1
            else:
                stats["failed"] += 1

        for row in updated_rows:
            success = self._upsert_and_ingest_object(row, operation="upsert")
            if success:
                stats["updated"] += 1
            else:
                stats["failed"] += 1

        for row in deleted_rows:
            success = self._handle_deleted_object(row)
            if success:
                stats["deleted"] += 1
            else:
                stats["failed"] += 1

        return stats

    def _upsert_and_ingest_object(self, row: Dict[str, Any], operation: str) -> bool:
        object_path = str(row.get("object_path") or "").strip()
        folder_key = str(row.get("folder_key") or "").strip()
        file_name = str(row.get("name") or os.path.basename(object_path) or "document")

        if not object_path or not folder_key:
            return False

        collection_name = build_collection_name(folder_key)
        source_updated_at = str(row.get("updated_at") or "").strip()
        source_etag = str(row.get("etag") or "").strip() or None
        content_type = str(row.get("content_type") or "application/octet-stream")
        size = int(row.get("size") or 0)

        temp_path = None
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.object_path == object_path).first()
            if document is None:
                document = Document(
                    original_name=file_name,
                    stored_name=file_name,
                    path=object_path,
                    object_path=object_path,
                    folder_key=folder_key,
                    collection_name=collection_name,
                    mime_type=content_type,
                    size=size,
                    status="pending",
                    total_chunks=0,
                    last_synced_at=utcnow(),
                )
                db.add(document)
            else:
                document.original_name = file_name
                document.stored_name = file_name
                document.path = object_path
                document.folder_key = folder_key
                document.collection_name = collection_name
                document.mime_type = content_type
                document.size = size
                document.status = "pending"
                document.deleted_at = None
                document.last_synced_at = utcnow()

            document.source_etag = source_etag
            parsed_updated_at = _parse_iso_datetime(source_updated_at)
            if parsed_updated_at is not None:
                document.source_updated_at = parsed_updated_at

            db.commit()
            db.refresh(document)
            document_id = document.id
        except Exception as error:
            db.rollback()
            log_document_sync_error(
                db,
                object_path=object_path,
                operation=operation,
                error_message=str(error),
                folder_key=folder_key,
                collection_name=collection_name,
                payload=row,
            )
            return False
        finally:
            db.close()

        try:
            temp_path = self.sync_service.download_object(object_path)
            success = process_document_ingest(
                document_id=document_id,
                file_path=temp_path,
                collection_name=collection_name,
                source_path=object_path,
                source_object_path=object_path,
                source_updated_at=source_updated_at,
                source_etag=source_etag,
                cleanup_file=True,
                size=size,
            )

            db = SessionLocal()
            try:
                if success:
                    mark_document_sync_error_resolved(
                        db,
                        object_path=object_path,
                        operation=operation,
                    )
                    return True

                log_document_sync_error(
                    db,
                    object_path=object_path,
                    operation=operation,
                    error_message="Ingest failed for synced object",
                    folder_key=folder_key,
                    collection_name=collection_name,
                    payload=row,
                )
                return False
            finally:
                db.close()
        except Exception as error:
            db = SessionLocal()
            try:
                log_document_sync_error(
                    db,
                    object_path=object_path,
                    operation=operation,
                    error_message=str(error),
                    folder_key=folder_key,
                    collection_name=collection_name,
                    payload=row,
                )
            finally:
                db.close()

            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    logger.exception("Failed to remove temporary sync file: %s", temp_path)
            return False

    def _handle_deleted_object(self, row: Dict[str, Any]) -> bool:
        object_path = str(row.get("object_path") or "").strip()
        folder_key = str(row.get("folder_key") or "").strip()
        if not object_path:
            return False

        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.object_path == object_path).first()
            collection_name = ""
            if document is not None:
                collection_name = str(document.collection_name or "").strip()
            if not collection_name and folder_key:
                collection_name = build_collection_name(folder_key)

            if collection_name:
                try:
                    delete_vectors_for_object_path(collection_name=collection_name, object_path=object_path)
                except Exception:
                    logger.exception("Failed deleting vectors for object_path=%s", object_path)

            if document is not None:
                document.deleted_at = utcnow()
                document.last_synced_at = utcnow()
                document.status = "deleted"
                document.error_message = None
                db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()

            db.commit()

            mark_document_sync_error_resolved(
                db,
                object_path=object_path,
                operation="delete",
            )
            return True
        except Exception as error:
            db.rollback()
            log_document_sync_error(
                db,
                object_path=object_path,
                operation="delete",
                error_message=str(error),
                folder_key=folder_key or None,
                collection_name=build_collection_name(folder_key) if folder_key else None,
                payload=row,
            )
            return False
        finally:
            db.close()


async def run_supabase_sync_scheduler(
    sync_service: SupabaseStorageSyncService,
    interval_seconds: int,
    stop_event: asyncio.Event,
) -> None:
    logger.info("Supabase sync scheduler started. interval_seconds=%s", interval_seconds)

    while not stop_event.is_set():
        try:
            result = await asyncio.to_thread(sync_service.scan_and_diff_snapshot)
            added_count = len(result.get("added", []))
            updated_count = len(result.get("updated", []))
            deleted_count = len(result.get("deleted", []))

            if added_count or updated_count or deleted_count:
                logger.info(
                    "Supabase sync changed: added=%s updated=%s deleted=%s total_objects=%s",
                    added_count,
                    updated_count,
                    deleted_count,
                    result.get("total_objects", 0),
                )
            else:
                logger.debug("Supabase sync: no change. total_objects=%s", result.get("total_objects", 0))

            try:
                await asyncio.to_thread(_resolve_scheduler_sync_error)
            except Exception:
                logger.exception("Failed to resolve scheduler sync error state.")
        except Exception as error:
            logger.exception("Supabase sync scheduler iteration failed.")
            try:
                await asyncio.to_thread(_persist_scheduler_sync_error, str(error))
            except Exception:
                logger.exception("Failed to persist scheduler sync error state.")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue

    logger.info("Supabase sync scheduler stopped.")


def _persist_scheduler_sync_error(error_message: str) -> None:
    db = SessionLocal()
    try:
        log_document_sync_error(
            db,
            object_path=_SCHEDULER_OBJECT_PATH,
            operation=_SCHEDULER_OPERATION,
            error_message=error_message,
            payload={"scope": "supabase_sync_scheduler"},
        )
    finally:
        db.close()


def _resolve_scheduler_sync_error() -> None:
    db = SessionLocal()
    try:
        mark_document_sync_error_resolved(
            db,
            object_path=_SCHEDULER_OBJECT_PATH,
            operation=_SCHEDULER_OPERATION,
        )
    finally:
        db.close()
