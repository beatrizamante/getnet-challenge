# pylint: disable=redefined-outer-name,protected-access
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.infrastructure.adapters.embeddings.huggingface_adapter import HuggingFaceEmbeddingAdapter
from src.infrastructure.config.settings import EmbeddingSettings

_DIM = 384  # all-MiniLM-L6-v2 output dimension
_SETTINGS = EmbeddingSettings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _make_mock_model(n_results: int = 1) -> MagicMock:
    mock = MagicMock()
    mock.encode.return_value = np.random.rand(n_results, _DIM).astype(np.float32)
    return mock


@pytest.fixture
def adapter_with_mock_model():
    mock_model = _make_mock_model()
    with patch(
        "src.infrastructure.adapters.embeddings.huggingface_adapter.SentenceTransformer",
        return_value=mock_model,
    ):
        adapter = HuggingFaceEmbeddingAdapter(_SETTINGS)
        yield adapter, mock_model


async def test_embed_returns_list_of_floats(adapter_with_mock_model):
    adapter, _ = adapter_with_mock_model
    result = await adapter.embed("test text")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


async def test_embed_output_matches_model_dimension(adapter_with_mock_model):
    adapter, _ = adapter_with_mock_model
    result = await adapter.embed("test text")
    assert len(result) == _DIM


async def test_embed_calls_encode_with_normalize(adapter_with_mock_model):
    adapter, mock_model = adapter_with_mock_model
    await adapter.embed("hello")
    _, call_kwargs = mock_model.encode.call_args
    assert call_kwargs.get("normalize_embeddings") is True


async def test_embed_batch_returns_correct_shape(adapter_with_mock_model):
    adapter, mock_model = adapter_with_mock_model
    texts = ["text one", "text two", "text three"]
    mock_model.encode.return_value = np.random.rand(len(texts), _DIM).astype(np.float32)
    result = await adapter.embed_batch(texts)
    assert len(result) == 3
    assert all(len(row) == _DIM for row in result)


async def test_embed_batch_single_encode_call(adapter_with_mock_model):
    adapter, mock_model = adapter_with_mock_model
    texts = ["a", "b", "c"]
    mock_model.encode.return_value = np.random.rand(len(texts), _DIM).astype(np.float32)
    await adapter.embed_batch(texts)
    # Native batching: encode must be called exactly once for all texts
    mock_model.encode.assert_called_once()


async def test_embed_batch_empty_input_returns_empty(adapter_with_mock_model):
    adapter, mock_model = adapter_with_mock_model
    result = await adapter.embed_batch([])
    assert result == []
    mock_model.encode.assert_not_called()


async def test_model_loaded_lazily():
    """SentenceTransformer must not be instantiated at adapter creation time."""
    with patch(
        "src.infrastructure.adapters.embeddings.huggingface_adapter.SentenceTransformer"
    ) as MockST:
        _ = HuggingFaceEmbeddingAdapter(_SETTINGS)
        MockST.assert_not_called()


async def test_model_loaded_only_once(adapter_with_mock_model):
    """Multiple calls must share a single SentenceTransformer instance."""
    adapter, mock_model = adapter_with_mock_model
    mock_model.encode.return_value = np.random.rand(1, _DIM).astype(np.float32)
    await adapter.embed("first")
    await adapter.embed("second")
    # Both calls go through the same model; ST constructor must only have fired once
    assert adapter._model is mock_model
