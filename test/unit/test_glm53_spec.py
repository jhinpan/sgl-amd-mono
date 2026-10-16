"""The GLM-5.3 gfx950 monokernel spec on CPU (RFC #43949): the layers its forward
covers on the GLM-5.3 layer structure, its row routing, its registry entry, and its
config and model refusal reasons. SGLang checks its declared requirements, the same
way for every spec."""

from types import SimpleNamespace

import torch
from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.glm53 import HF_CONFIG
from sgl_amd_mono.glm53 import spec as glm53

# GLM-5.3's layer structure and quantization (its config.json)
GLM53 = SimpleNamespace(
    **HF_CONFIG,
    num_hidden_layers=78,
    first_k_dense_replace=3,
    moe_layer_freq=1,
    quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
    architectures=["GlmMoeDsaForCausalLM"],
)


def test_the_forward_covers_the_moe_layers():
    # the dense layers 0 to 2 run SGLang's path inside the forward
    assert glm53._covered_layers(GLM53) == list(range(3, 78))


def test_the_plugin_registers_the_spec():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    arch = ["GlmMoeDsaForCausalLM"]
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=arch, isa="gfx950"
    )
    assert found is glm53.SPEC, why
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=arch, isa="gfx942"
    )
    assert found is None and "gfx942" in why


def test_a_step_runs_at_the_next_width():
    # a one-row launch never completes on gfx950: one row runs at width 2
    assert [glm53._width(r) for r in glm53.ROWS] == [2, 2, 4, 4, 8, 8, 8, 8]


def test_config_reasons_name_each_mismatch():
    cfg = SimpleNamespace(page_size=None)
    assert glm53._config_disable_reasons(cfg, GLM53) == []
    other = SimpleNamespace(
        **{
            **vars(GLM53),
            "index_topk_freq": 8,
            "moe_layer_freq": 2,
            "quantization_config": {"quant_method": "quark"},
        }
    )
    reasons = glm53._config_disable_reasons(SimpleNamespace(page_size=1), other)
    text = " ".join(reasons)
    assert "index_topk_freq" in text
    assert "MoE" in text
    assert "--page-size 1" in text
    assert "FP8 block-128" in text


def _layer(experts_dtype=torch.float8_e4m3fn, n_experts=257, indexer=True):
    w13 = torch.empty(n_experts, 1, 1, dtype=experts_dtype)
    return SimpleNamespace(
        mlp=SimpleNamespace(experts=SimpleNamespace(w13_weight=w13)),
        self_attn=SimpleNamespace(indexer=object() if indexer else None),
    )


def test_model_reasons():
    def model(layers):
        return SimpleNamespace(model=SimpleNamespace(layers=layers))

    good = [_layer(indexer=i % 4 == 2 or i < 3) for i in range(78)]
    assert glm53._model_disable_reasons(model(good), GLM53, {}) == []
    bad = list(good)
    bad[5] = _layer(experts_dtype=torch.bfloat16)
    bad[9] = _layer(n_experts=256)
    bad[2] = _layer(indexer=False)
    reasons = " ".join(glm53._model_disable_reasons(model(bad), GLM53, {}))
    assert "layer 5: needs block-FP8 experts" in reasons
    assert "layer 9: needs the shared expert fused" in reasons
    assert "layer 3 reuses the top-k of layer 2" in reasons
