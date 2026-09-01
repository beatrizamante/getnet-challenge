import asyncio
import logging
from dataclasses import dataclass

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src.infrastructure.adapters.llm.deepseek_judge import DeepSeekJudgeModel
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter
from src.domain.shared.constants import CONTEXT_CHUNK_SEPARATOR

logger = logging.getLogger(__name__)


@dataclass
class OutputGuardrailResult:
    passed: bool
    score: float | None  # None means evaluation was skipped (timeout or error)
    reason: str = ""
    disclaimer_added: bool = False


class OutputGuardrail:
    """Post-generation faithfulness check using DeepEval with an independent judge model.

    The judge (DeepSeekJudgeModel) is intentionally different from the response model
    to avoid self-evaluation bias — a model cannot reliably judge its own output.
    """

    def __init__(
        self,
        judge: DeepSeekJudgeModel,
        langfuse: LangfuseAdapter,
        threshold: float = 0.7,
        judge_timeout: float = 10.0,
        disclaimer: str = "\n\n⚠️ Note: this response may not be fully grounded in Getnet's official documentation. Please verify with official support.",
    ) -> None:
        self._metric = FaithfulnessMetric(
            threshold=threshold,
            model=judge,
            include_reason=True,
        )
        self._langfuse = langfuse
        self._threshold = threshold
        self._judge_timeout = judge_timeout
        self._disclaimer = disclaimer

    async def check(
        self,
        question: str,
        answer: str,
        context: str,
        trace_id: str = "",
    ) -> OutputGuardrailResult:
        """Evaluate faithfulness; returns result. Never raises — fails open on timeout/error."""
        if not context.strip():
            return OutputGuardrailResult(passed=True, score=1.0, reason="no context to evaluate")

        chunks = _split_context(context)
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=chunks,  # type: ignore[arg-type]
        )
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._metric.measure, test_case),
                timeout=self._judge_timeout,
            )
            score: float = self._metric.score or 0.0
            reason: str = self._metric.reason or ""
            passed = score >= self._threshold

            self._langfuse.score(trace_id, "faithfulness", score)
            logger.info("Output guardrail: score=%.2f passed=%s reason=%s", score, passed, reason)
            return OutputGuardrailResult(passed=passed, score=score, reason=reason)

        except asyncio.TimeoutError:
            logger.warning("Output guardrail timed out after %.1fs — passing through.", self._judge_timeout)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Output guardrail failed — passing through. error=%s", exc)

        return OutputGuardrailResult(passed=True, score=None, reason="evaluation skipped")

    async def apply(self, question: str, answer: str, context: str, trace_id: str = "") -> str:
        """Run the check and return the answer (with disclaimer appended if flagged)."""
        result = await self.check(question, answer, context, trace_id)
        if not result.passed and result.score is not None:
            return answer + self._disclaimer
        return answer


def _split_context(context: str) -> list[str]:
    """Recover individual source chunks from the formatted context string."""
    parts = [p.strip() for p in context.split(CONTEXT_CHUNK_SEPARATOR)]
    return [p for p in parts if p]
