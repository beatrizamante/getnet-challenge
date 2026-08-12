import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from src.domain.shared.Agent_State import AgentState

from src.application.guardrails.output_guardrail import _split_context
from src.application.guardrails.input_guardrail import InputGuardrail
from src.infrastructure.adapters.llm.deepseek_judge import DeepSeekJudgeModel

logger = logging.getLogger(__name__)

JOB_NAME = "run_eval_suite"

_DATASET_PATH = Path(__file__).parent.parent.parent / "__tests__" / "evaluation" / "golden_dataset.json"

_THRESHOLDS = {
    "faithfulness": 0.8,
    "answer_relevancy": 0.7,
    "contextual_relevancy": 0.7,
    "contextual_precision": 0.7,
    "contextual_recall": 0.7,
    "routing_accuracy": 0.90,
}

_METRIC_TIMEOUT = 300.0
_EVAL_REDIS_KEY = "eval:latest"


async def run_eval_suite(ctx: dict[str, Any]) -> dict[str, Any]:
    """ARQ job: run the full DeepEval suite against the golden dataset and store results in Redis."""
    container = ctx["container"]
    graph = await container.agent_graph.async_()
    redis_client = container.redis_client()
    input_guard: InputGuardrail = container.input_guardrail()
    langfuse = container.langfuse_adapter()

    settings = container.settings()
    judge = DeepSeekJudgeModel(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        model_name=settings.guardrail.model,
    )

    dataset = _load_dataset()
    logger.info("Eval suite started. entries=%d", len(dataset))

    faithfulness_m = FaithfulnessMetric(threshold=_THRESHOLDS["faithfulness"], model=judge)
    answer_rel_m = AnswerRelevancyMetric(threshold=_THRESHOLDS["answer_relevancy"], model=judge)
    ctx_rel_m = ContextualRelevancyMetric(threshold=_THRESHOLDS["contextual_relevancy"], model=judge)
    ctx_prec_m = ContextualPrecisionMetric(threshold=_THRESHOLDS["contextual_precision"], model=judge)
    ctx_recall_m = ContextualRecallMetric(threshold=_THRESHOLDS["contextual_recall"], model=judge)

    per_entry: list[dict] = []
    routing_correct = 0
    routing_total = 0

    for entry in dataset:
        entry_result = await _eval_entry(
            entry=entry,
            graph=graph,
            input_guard=input_guard,
            faithfulness_m=faithfulness_m,
            answer_rel_m=answer_rel_m,
            ctx_rel_m=ctx_rel_m,
            ctx_prec_m=ctx_prec_m,
            ctx_recall_m=ctx_recall_m,
        )
        per_entry.append(entry_result)

        if entry_result.get("routing_match") is not None:
            routing_total += 1
            if entry_result["routing_match"]:
                routing_correct += 1

    routing_accuracy = routing_correct / routing_total if routing_total else 0.0

    summary = _aggregate(per_entry, routing_accuracy)
    summary["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    summary["entries"] = len(dataset)

    await redis_client.set(_EVAL_REDIS_KEY, json.dumps(summary))
    logger.info("Eval suite complete. summary=%s", summary)

    trace_id = langfuse.trace(JOB_NAME, summary)
    for metric, score in summary.get("scores", {}).items():
        langfuse.score(trace_id, metric, score)
    langfuse.flush()

    return summary


async def _eval_entry(
    entry: dict,
    graph,
    input_guard: InputGuardrail,
    **metrics,
) -> dict:
    result: dict[str, Any] = {"id": entry["id"], "input": entry["input"]}

    if entry.get("expected_intent") == "guardrail_blocked":
        guard = await input_guard.check(entry["input"])
        result["routing_match"] = guard.blocked
        result["blocked"] = guard.blocked
        return result

    state: AgentState = {"messages": [entry["input"]], "user_id": entry["user_id"]}
    graph_result = await graph.ainvoke(state)
    raw_response: dict = graph_result.get("response") or {}
    answer = str(raw_response.get("answer") or "")
    context = str(graph_result.get("context") or "")
    actual_route = str(graph_result.get("route") or raw_response.get("source_agent") or "")

    result["answer"] = answer
    result["route"] = actual_route
    result["routing_match"] = actual_route == entry.get("expected_intent")

    # --- only run RAG metrics on knowledge entries with context ---
    if entry.get("expected_intent") != "knowledge" or not context:
        return result

    chunks = _split_context(context)
    if not chunks:
        return result

    test_case = LLMTestCase(
        input=entry["input"],
        actual_output=answer,
        expected_output=entry.get("expected_output", ""),
        retrieval_context=chunks,  # type: ignore[arg-type]
    )

    for name, metric in [
        ("faithfulness", metrics["faithfulness_m"]),
        ("answer_relevancy", metrics["answer_rel_m"]),
        ("contextual_relevancy", metrics["ctx_rel_m"]),
        ("contextual_precision", metrics["ctx_prec_m"]),
        ("contextual_recall", metrics["ctx_recall_m"]),
    ]:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(metric.measure, test_case),
                timeout=_METRIC_TIMEOUT,
            )
            result[name] = metric.score
        except asyncio.TimeoutError:
            logger.warning("Metric %s timed out for entry %s", name, entry["id"])
            result[name] = None
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Metric %s failed for entry %s: %s", name, entry["id"], exc)
            result[name] = None

    return result


def _aggregate(per_entry: list[dict], routing_accuracy: float) -> dict:
    metric_keys = ["faithfulness", "answer_relevancy", "contextual_relevancy",
                   "contextual_precision", "contextual_recall"]
    scores: dict[str, float] = {}
    for key in metric_keys:
        values = [e[key] for e in per_entry if e.get(key) is not None]
        scores[key] = round(sum(values) / len(values), 4) if values else 0.0
    scores["routing_accuracy"] = round(routing_accuracy, 4)

    passed = all(scores.get(k, 0) >= v for k, v in _THRESHOLDS.items())
    return {"status": "pass" if passed else "fail", "scores": scores, "thresholds": _THRESHOLDS}


def _load_dataset() -> list[dict]:
    with open(_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)
