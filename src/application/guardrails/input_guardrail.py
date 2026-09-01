import logging
import re
from dataclasses import dataclass

from pydantic import BaseModel

from src.domain.ports.LLM_Port import LLMPort
from src.infrastructure.config.prompt_catalog import PromptCatalog, load_prompt_catalog

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all |your )?(previous |prior )?instructions",
        r"disregard (your |all )?instructions",
        r"forget (everything|your instructions|your prompt)",
        r"you are now\b",
        r"pretend (you are|to be)\b",
        r"act as (if you are|a )?\b",
        r"jailbreak",
        r"override (your |the )?system prompt",
        r"<\s*script\s*>",
        r"\{\{.*\}\}",
    ]
]


@dataclass
class InputGuardrailResult:
    blocked: bool
    reason: str = ""
    safe_response: str = ""


class _SafetyDecision(BaseModel):
    safe: bool
    reason: str


class InputGuardrail:
    """Pre-routing guardrail: rule-based injection detection + LLM safety classifier."""

    def __init__(
        self,
        llm: LLMPort,
        safe_rejection: str = "I'm unable to process that request. I'm here to help with questions about Getnet's payment solutions and services.",
        prompts: PromptCatalog | None = None,
    ) -> None:
        self._llm = llm
        self._safe_rejection = safe_rejection
        self._prompts = prompts or load_prompt_catalog()

    async def check(self, message: str) -> InputGuardrailResult:
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(message):
                logger.warning("Input blocked by rule. pattern=%s", pattern.pattern)
                return InputGuardrailResult(
                    blocked=True, reason="prompt_injection", safe_response=self._safe_rejection
                )

        try:
            decision = await self._llm.complete_structured(
                prompt=self._prompts.input_guardrail_classifier.format(message=message),
                schema=_SafetyDecision,
            )
            if not decision.safe:
                logger.warning("Input blocked by LLM classifier. reason=%s", decision.reason)
                return InputGuardrailResult(
                    blocked=True, reason=decision.reason, safe_response=self._safe_rejection
                )
        except Exception:
            try:
                raw = await self._llm.complete(
                    prompt=f"Is this message safe for a payment support chatbot? Reply only 'safe' or 'unsafe'.\nMessage: {message}",
                    system="",
                )
                if "unsafe" in raw.lower():
                    logger.warning("Input blocked by fallback LLM check.")
                    return InputGuardrailResult(
                        blocked=True, reason="llm_fallback", safe_response=self._safe_rejection
                    )
            except Exception as exc2:
                logger.warning("Input guardrail fully failed, passing through. error=%s", exc2)

        return InputGuardrailResult(blocked=False)
