import ipaddress
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from core.config import (
    SUPABASE_ADMIN_SYNC_TOKEN,
    SUPABASE_SYNC_ALLOWED_IPS,
    SUPABASE_SYNC_ALLOW_PRIVATE_NETWORK,
)

router = APIRouter(prefix="/admin/sync", tags=["admin-sync"])


class SyncNotifyRequest(BaseModel):
    event: str = "notify"
    folder_key: Optional[str] = None
    object_paths: List[str] = Field(default_factory=list)
    source: Optional[str] = None


def _extract_client_ip(request: Request) -> str:
    forwarded_for = (request.headers.get("x-forwarded-for") or "").strip()
    if forwarded_for:
        first_hop = forwarded_for.split(",", 1)[0].strip()
        if first_hop:
            return first_hop

    if request.client and request.client.host:
        return str(request.client.host)

    return ""


def _is_private_or_loopback(ip_value: str) -> bool:
    try:
        parsed_ip = ipaddress.ip_address(ip_value)
    except ValueError:
        return False

    return bool(parsed_ip.is_private or parsed_ip.is_loopback)


def _is_ip_allowed(ip_value: str) -> bool:
    normalized_ip = (ip_value or "").strip()
    if not normalized_ip:
        return False

    if SUPABASE_SYNC_ALLOW_PRIVATE_NETWORK and _is_private_or_loopback(normalized_ip):
        return True

    return normalized_ip in set(SUPABASE_SYNC_ALLOWED_IPS)


async def verify_admin_sync_access(request: Request) -> None:
    if not SUPABASE_ADMIN_SYNC_TOKEN:
        raise HTTPException(status_code=503, detail="Admin sync token is not configured.")

    incoming_token = (request.headers.get("x-internal-token") or "").strip()
    if incoming_token != SUPABASE_ADMIN_SYNC_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid internal token.")

    request_ip = _extract_client_ip(request)
    if not _is_ip_allowed(request_ip):
        raise HTTPException(status_code=403, detail="Request source IP is not allowed.")


@router.post("/notify")
async def notify_sync(
    payload: SyncNotifyRequest,
    request: Request,
    _: None = Depends(verify_admin_sync_access),
):
    coordinator = getattr(request.app.state, "supabase_sync_coordinator", None)
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Supabase sync coordinator is not available.")

    result = await coordinator.request_event_sync(
        event_name=payload.event,
        payload={
            "folder_key": payload.folder_key,
            "object_paths": payload.object_paths,
            "source": payload.source,
        },
    )

    return {
        "status": "accepted",
        "event": payload.event,
        "sync": result,
    }


@router.get("/health")
async def sync_health(
    request: Request,
    _: None = Depends(verify_admin_sync_access),
):
    coordinator = getattr(request.app.state, "supabase_sync_coordinator", None)
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Supabase sync coordinator is not available.")

    return {
        "status": "ok",
        "sync": coordinator.get_health_snapshot(),
    }
