"""
dashboard_routes.py
-------------------
REST API routes for the standalone Visitor Dashboard.

ENDPOINTS:
  GET  /api/dashboard/stats              → Key metrics (total visitors, active visits, today's meetings)
  GET  /api/dashboard/visitors           → All registered visitor profiles
  GET  /api/dashboard/meetings           → All meetings with host/visitor details
  GET  /api/dashboard/logs               → Reception log history (filterable)
  POST /api/dashboard/logs/{id}/checkout → Check out a visitor
  GET  /api/dashboard/visitors/{id}/photo → Serve visitor photo
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from receptionist.database import SessionLocal
from receptionist.models import Visitor, Meeting, ReceptionLog, Employee
from services.face_recognition_service import get_visitor_photo_path

logger = logging.getLogger(__name__)
router = APIRouter(tags=["dashboard"])


# ── DB session dependency ─────────────────────────────────────────────────────


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Pydantic response schemas ────────────────────────────────────────────────


class StatsOut(BaseModel):
    total_visitors: int
    active_visits: int
    todays_meetings: int
    pending_checkouts: int
    recent_checkins_today: int


class VisitorOut(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    has_photo: bool
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    total_visits: int

    class Config:
        from_attributes = True


class MeetingOut(BaseModel):
    id: int
    host_name: str
    host_department: Optional[str] = None
    visitor_name: Optional[str] = None
    visitor_id: Optional[int] = None
    scheduled_start: str
    scheduled_end: Optional[str] = None
    purpose: Optional[str] = None
    status: str
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


class LogOut(BaseModel):
    id: int
    visitor_name: Optional[str] = None
    visitor_id: Optional[int] = None
    employee_name: Optional[str] = None
    person_type: str
    badge_id: Optional[str] = None
    check_in_time: Optional[str] = None
    check_out_time: Optional[str] = None
    purpose: Optional[str] = None
    is_active: bool

    class Config:
        from_attributes = True


# ── Helper: format datetime safely ───────────────────────────────────────────


def _fmt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/stats", response_model=StatsOut)
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Dashboard overview statistics."""
    total_visitors = db.query(func.count(Visitor.id)).scalar() or 0

    active_visits = (
        db.query(func.count(ReceptionLog.id))
        .filter(ReceptionLog.check_out_time.is_(None))
        .scalar()
        or 0
    )

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)

    todays_meetings = (
        db.query(func.count(Meeting.id))
        .filter(
            Meeting.scheduled_start >= today_start,
            Meeting.scheduled_start < today_end,
        )
        .scalar()
        or 0
    )

    pending_checkouts = (
        db.query(func.count(ReceptionLog.id))
        .filter(
            ReceptionLog.check_out_time.is_(None),
            ReceptionLog.check_in_time >= today_start,
        )
        .scalar()
        or 0
    )

    recent_checkins_today = (
        db.query(func.count(ReceptionLog.id))
        .filter(
            ReceptionLog.check_in_time >= today_start,
            ReceptionLog.check_in_time < today_end,
        )
        .scalar()
        or 0
    )

    return StatsOut(
        total_visitors=total_visitors,
        active_visits=active_visits,
        todays_meetings=todays_meetings,
        pending_checkouts=pending_checkouts,
        recent_checkins_today=recent_checkins_today,
    )


@router.get("/visitors", response_model=List[VisitorOut])
def list_visitors(
    search: Optional[str] = Query(None, description="Search by name or email"),
    db: Session = Depends(get_db),
):
    """List all registered visitors with visit counts."""
    query = db.query(Visitor)

    if search:
        query = query.filter(
            Visitor.name.ilike(f"%{search}%") | Visitor.email.ilike(f"%{search}%")
        )

    visitors = query.order_by(Visitor.last_seen.desc().nullslast()).all()

    result = []
    for v in visitors:
        visit_count = (
            db.query(func.count(ReceptionLog.id))
            .filter(ReceptionLog.visitor_id == v.id)
            .scalar()
            or 0
        )
        has_photo = get_visitor_photo_path(v.id).exists()

        result.append(
            VisitorOut(
                id=v.id,
                name=v.name,
                email=v.email,
                phone=v.phone,
                has_photo=has_photo,
                first_seen=_fmt(v.first_seen),
                last_seen=_fmt(v.last_seen),
                total_visits=visit_count,
            )
        )
    return result


@router.get("/meetings", response_model=List[MeetingOut])
def list_meetings(
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search by host or visitor name"),
    db: Session = Depends(get_db),
):
    """List all meetings with host and visitor details."""
    query = (
        db.query(Meeting, Employee, Visitor)
        .join(Employee, Meeting.host_employee_id == Employee.id)
        .outerjoin(Visitor, Meeting.visitor_id == Visitor.id)
    )

    if status:
        query = query.filter(Meeting.status == status)

    if search:
        query = query.filter(
            Employee.name.ilike(f"%{search}%") | Visitor.name.ilike(f"%{search}%")
        )

    rows = query.order_by(Meeting.scheduled_start.desc()).all()

    return [
        MeetingOut(
            id=m.id,
            host_name=e.name,
            host_department=e.department,
            visitor_name=v.name if v else None,
            visitor_id=v.id if v else None,
            scheduled_start=_fmt(m.scheduled_start),
            scheduled_end=_fmt(m.scheduled_end),
            purpose=m.purpose,
            status=m.status,
            created_at=_fmt(m.created_at),
        )
        for m, e, v in rows
    ]


@router.get("/logs", response_model=List[LogOut])
def list_logs(
    active_only: bool = Query(False, description="Only show currently checked-in"),
    search: Optional[str] = Query(None, description="Search by name or badge"),
    db: Session = Depends(get_db),
):
    """List reception log history."""
    query = (
        db.query(ReceptionLog, Visitor, Employee)
        .outerjoin(Visitor, ReceptionLog.visitor_id == Visitor.id)
        .outerjoin(Employee, ReceptionLog.employee_id == Employee.id)
    )

    if active_only:
        query = query.filter(ReceptionLog.check_out_time.is_(None))

    if search:
        query = query.filter(
            Visitor.name.ilike(f"%{search}%")
            | Employee.name.ilike(f"%{search}%")
            | ReceptionLog.badge_id.ilike(f"%{search}%")
        )

    rows = query.order_by(ReceptionLog.check_in_time.desc()).all()

    return [
        LogOut(
            id=log.id,
            visitor_name=v.name if v else None,
            visitor_id=v.id if v else None,
            employee_name=e.name if e else None,
            person_type=log.person_type,
            badge_id=log.badge_id,
            check_in_time=_fmt(log.check_in_time),
            check_out_time=_fmt(log.check_out_time),
            purpose=log.purpose,
            is_active=log.check_out_time is None,
        )
        for log, v, e in rows
    ]


@router.post("/logs/{log_id}/checkout")
def checkout_log(log_id: int, db: Session = Depends(get_db)):
    """Check out a visitor by log ID."""
    entry = db.query(ReceptionLog).filter(ReceptionLog.id == log_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail=f"Log entry {log_id} not found")
    if entry.check_out_time is not None:
        raise HTTPException(status_code=400, detail="Already checked out")

    entry.check_out_time = datetime.utcnow()
    db.commit()

    logger.info(
        "Dashboard checkout: log_id=%d checked out at %s", log_id, entry.check_out_time
    )
    return {
        "success": True,
        "log_id": log_id,
        "check_out_time": _fmt(entry.check_out_time),
    }


@router.get("/visitors/{visitor_id}/photo")
def serve_visitor_photo(visitor_id: int, db: Session = Depends(get_db)):
    """Serve the stored visitor face photo."""
    visitor = db.query(Visitor).filter(Visitor.id == visitor_id).first()
    if not visitor:
        raise HTTPException(status_code=404, detail=f"Visitor {visitor_id} not found")

    photo_path = get_visitor_photo_path(visitor_id)
    if not photo_path.exists():
        raise HTTPException(status_code=404, detail="No photo found for this visitor")

    return FileResponse(
        path=str(photo_path),
        media_type="image/jpeg",
        filename=f"visitor_{visitor_id}.jpg",
    )
