from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from app.errors import InvalidInputError, ModelInferenceError  # noqa: E402
from app.inference.generation import generate_prompts, synchronize  # noqa: E402


class Encoding(dict):
    def to(self, device):
        return self


class Tokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, text, **kwargs):
        self.seen_padding = self.padding_side
        rows = [[1, 2], [0, 3]] if isinstance(text, list) else [[1, 2]]
        ids = torch.tensor(rows)
        return Encoding(input_ids=ids, attention_mask=(ids != 0).long())

    def decode(self, ids, **kwargs):
        return str(ids.tolist())


def loaded():
    model = Mock()

    def generate(**kwargs):
        assert torch.is_inference_mode_enabled()
        assert kwargs["use_cache"] is True
        assert kwargs["do_sample"] is False
        assert kwargs["num_beams"] == 1
        if streamer := kwargs.get("streamer"):
            streamer.put(kwargs["input_ids"])
            streamer.put(torch.tensor([10]))
            streamer.end()
        count = len(kwargs["input_ids"])
        return torch.cat((kwargs["input_ids"], torch.tensor([[10, 11]] * count)), dim=1)

    model.generate.side_effect = generate
    return SimpleNamespace(model=model, tokenizer=Tokenizer(), device="cpu")


def test_generation_inference_mode_and_stage_timing():
    model = loaded()
    text, timing = generate_prompts(model, ["prompt"], 32)[0]
    assert text == "[10, 11]"
    assert timing.prompt_tokens == timing.generated_tokens == 2
    assert timing.total_ms >= timing.latency_ms
    assert timing.first_token_ms is None
    assert not torch.is_inference_mode_enabled()


def test_optional_first_token_profiler():
    text, timing = generate_prompts(loaded(), ["prompt"], 32, profile_tokens=True)[0]
    assert text == "[10, 11]"
    assert timing.first_token_ms is not None
    assert 0 <= timing.first_token_ms <= timing.latency_ms


def test_batch_left_padding_is_temporary():
    model = loaded()
    results = generate_prompts(model, ["first", "second"], 32)
    assert model.tokenizer.seen_padding == "left"
    assert model.tokenizer.padding_side == "right"
    assert [timing.prompt_tokens for _, timing in results] == [2, 1]
    assert [text for text, _ in results] == ["[10, 11]", "[10, 11]"]


def test_batch_padding_restored_if_tokenizer_fails():
    model = loaded()
    model.tokenizer = Mock(padding_side="right", side_effect=ValueError("tokenizer"))
    with pytest.raises(ValueError, match="tokenizer"):
        generate_prompts(model, ["first", "second"], 32)
    assert model.tokenizer.padding_side == "right"


@pytest.mark.parametrize("prompts,limit", [([], None), (["prompt"], 1)])
def test_invalid_input_does_not_generate(prompts, limit):
    model = loaded()
    with pytest.raises(InvalidInputError):
        generate_prompts(model, prompts, 32, max_input_tokens=limit)
    model.model.generate.assert_not_called()


def test_model_error_becomes_controlled_error():
    model = loaded()
    model.model.generate.side_effect = RuntimeError("private runtime details")
    with pytest.raises(ModelInferenceError) as error:
        generate_prompts(model, ["prompt"], 32)
    assert "private" not in error.value.message
    assert not torch.is_inference_mode_enabled()


def test_device_synchronization(monkeypatch):
    cuda = Mock()
    mps = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", cuda)
    monkeypatch.setattr(torch.mps, "synchronize", mps)
    synchronize("cpu")
    cuda.assert_not_called()
    mps.assert_not_called()
    synchronize("cuda:0")
    synchronize("mps")
    cuda.assert_called_once()
    mps.assert_called_once()
