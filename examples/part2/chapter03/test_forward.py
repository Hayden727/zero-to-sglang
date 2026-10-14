"""CPU 离线验证：同权重参考实现、因果性、生成边界。"""

from dataclasses import asdict

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from .forward import Config, Qwen3, generate


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(123)
    torch.set_num_threads(1)


@pytest.mark.parametrize("tied", [False, True])
def test_logits_match_transformers(tied):
    # hidden_size 不必等于 num_attention_heads * head_dim。
    config = Config(hidden_size=24, tie_word_embeddings=tied)
    reference_config = Qwen3Config(**asdict(config), attention_dropout=0.0)
    reference_config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(reference_config).eval()
    model = Qwen3(config).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    with torch.inference_mode():
        expected = reference(ids, use_cache=False).logits
        torch.testing.assert_close(model(ids), expected, atol=2e-6, rtol=2e-5)
    if tied:
        assert model.lm_head.weight is model.model.embed_tokens.weight


def test_future_tokens_do_not_change_prefix():
    model = Qwen3().eval()
    with torch.inference_mode():
        first = model(torch.tensor([[1, 2, 3, 4]]))
        second = model(torch.tensor([[1, 2, 55, 56]]))
    torch.testing.assert_close(first[:, :2], second[:, :2])


def test_generation_matches_reference():
    config = Qwen3Config(**asdict(Config()), attention_dropout=0.0)
    config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(config).eval()
    model = Qwen3().eval()
    model.load_state_dict(reference.state_dict())
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = reference.generate(ids, max_new_tokens=8, do_sample=False,
                                      use_cache=False, pad_token_id=0, eos_token_id=None)
    assert generate(model, [1, 2, 3]) == expected[0, 3:].tolist()


def test_eos_and_length_boundaries():
    model = Qwen3().eval()
    first = generate(model, [1, 2], 1)[0]
    assert generate(model, [1, 2], 8, eos_id=first) == [first]
    assert generate(model, [1, 2], 0) == []
    assert len(generate(model, [1, 2], 5, temperature=0.8)) == 5
    with pytest.raises(ValueError, match="context"):
        generate(model, [1], 256)
    with pytest.raises(ValueError, match="temperature"):
        generate(model, [1], temperature=float("nan"))
