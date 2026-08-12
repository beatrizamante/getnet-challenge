from fastapi import APIRouter, Depends

from src._lib.container import Container, get_container
from src.application.agents.escalation_agent import EscalationAgent
from src.application.jobs.ingestion_job import JOB_NAME
from src.domain.ports.Queue_Port import QueuePort

router = APIRouter(prefix="/admin", tags=["admin"])


def _queue(container: Container = Depends(get_container)) -> QueuePort:
    return container.queue_port()


def _escalation(container: Container = Depends(get_container)) -> EscalationAgent:
    return container.escalation_agent()


@router.post("/ingest")
async def trigger_ingest(force: bool = False, queue: QueuePort = Depends(_queue)) -> dict:
    """Enqueue the Getnet web-scraping & ingestion job. Returns immediately with the job ID."""
    job_id = await queue.enqueue(JOB_NAME, force=force)
    return {"status": "queued", "job": JOB_NAME, "job_id": job_id}


@router.get("/escalations/{user_id}")
async def get_escalations(
    user_id: str, escalation: EscalationAgent = Depends(_escalation)
) -> dict:
    """Return the escalation audit log for a user (most recent first)."""
    events = await escalation.get_audit_log(user_id)
    return {"user_id": user_id, "count": len(events), "events": events}
