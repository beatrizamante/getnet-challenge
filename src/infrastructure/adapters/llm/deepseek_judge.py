import asyncio
import re
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI


class DeepSeekJudgeModel(DeepEvalBaseLLM):
    """DeepEval-compatible wrapper around a DeepSeek (OpenAI-compatible) model.

    Used exclusively as the LLM-as-judge for output faithfulness scoring —
    intentionally a DIFFERENT model from the one that generated the response.
    """

    def __init__(self, api_key: str, base_url: str, model_name: str) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model_name = model_name
        super().__init__(model_name)

    def load_model(self, *args: Any, **kwargs: Any) -> Any:  # pyright: ignore[reportReturnType]
        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def generate(self, *args: Any, **kwargs: Any) -> str:
        prompt: str = args[0] if args else str(kwargs.get("prompt", ""))
        client: OpenAI = self.load_model()
        response = client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},  # disable thinking mode for faster judge calls
        )
        content = response.choices[0].message.content or ""
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    async def a_generate(self, *args: Any, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.generate, *args, **kwargs)

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self._model_name
