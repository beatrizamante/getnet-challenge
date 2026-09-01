import json
from functools import lru_cache
from pathlib import Path

from src.domain.shared.Prompt_Catalog import PromptCatalog

_DEFAULT_PROMPTS_PATH = Path(__file__).with_name("prompts.json")

@lru_cache(maxsize=None)
def load_prompt_catalog(prompts_file: str = "") -> PromptCatalog:
    path = Path(prompts_file) if prompts_file else _DEFAULT_PROMPTS_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load prompt catalog from {path}") from exc
    return PromptCatalog.model_validate(raw)
