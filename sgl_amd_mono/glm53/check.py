# SPDX-License-Identifier: Apache-2.0
"""SGLANG_AMD_MONO_CHECK=1 (eager only): each covered layer also runs SGLang's
layer on its input, on the caches as they were, and checks the kernel's outputs
against it: the hidden stream it writes, and the cache rows the layer writes, each
against its ``spec.NUMERICS`` bound. Every check is logged; a miss is logged as a
violation."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/layer_boundary/residual/batch.py": (
        "fold",
        "take_output",
        "set_written",
        "written_residual",
    ),
    "srt/mem_cache/memory_pool.py": ("get_key_buffer", "get_index_k_with_scale_buffer"),
    "srt/model_executor/forward_batch_info.py": ("out_cache_loc",),
    "srt/model_executor/forward_context.py": ("get_token_to_kv_pool",),
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"'),
    "srt/utils/common.py": ("BumpAllocator",),
}

PAGE_TOKENS = 64  # tokens an index-cache page covers
INDEX_VALUE_BYTES = PAGE_TOKENS * 128  # an index page's FP8 values; f32 scales follow


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    if a.numel() == 0:
        return {}
    return {
        "cosine": F.cosine_similarity(a, b, dim=0).item(),
        "rel": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
        "max_abs": (a - b).abs().max().item(),
    }


def _outputs(name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, dict]:
    if name != "index":
        return {name: _metrics(a, b)}
    # both pages in one layout: compare the decoded FP8 values and the f32 scales
    fp8 = torch.float8_e4m3fn
    return {
        "index": _metrics(
            a[:, :INDEX_VALUE_BYTES].contiguous().view(fp8),
            b[:, :INDEX_VALUE_BYTES].contiguous().view(fp8),
        ),
        "index_scale": _metrics(
            a[:, INDEX_VALUE_BYTES:].contiguous().view(torch.float32),
            b[:, INDEX_VALUE_BYTES:].contiguous().view(torch.float32),
        ),
    }


def _regions(L, forward_batch, S: int) -> dict:
    """The cache entries the layer writes this step: name -> (tensor, index)."""
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    pool = get_token_to_kv_pool()
    loc = forward_batch.out_cache_loc[:S]
    slots = loc[loc != 0].long()
    kv = pool.get_key_buffer(L.i)
    regions = {"kv": (kv.view(kv.shape[0], -1), slots)}
    if L.w.index is not None:
        regions["index"] = (
            pool.get_index_k_with_scale_buffer(L.i),
            (slots // PAGE_TOKENS).unique(),
        )
    return regions


def reference(L, positions, h_in, forward_batch, S: int, prev_topk) -> dict:
    """SGLang's layer on ``h_in`` [S, HIDDEN] (a copy): its output stream, its
    top-k and the cache rows it writes; the caches are restored after."""
    from sglang.srt.layers.layer_boundary.residual.batch import (
        fold,
        set_written,
        take_output,
        written_residual,
    )
    from sglang.srt.utils.common import BumpAllocator

    regions = _regions(L, forward_batch, S)
    saved = {name: t[idx].clone() for name, (t, idx) in regions.items()}
    hidden = set_written(h_in, forward_batch)
    zero = BumpAllocator(buffer_size=4, dtype=torch.float32, device=h_in.device)
    # the stream after the attention half (h + attention), as the FFN stage sees it
    seen = {}
    def ffn_in(mod, args):
        seen.setdefault("attn", written_residual(forward_batch).clone())
        x = args[0] if args else None
        if torch.is_tensor(x):
            seen.setdefault("ffn_in", x.clone())
            w = mod.gate.weight.float()
            seen.setdefault("scores_torch", torch.sigmoid(x.float() @ w.t()))
            bias = mod.gate.e_score_correction_bias.float()
            choice = seen["scores_torch"] + bias
            seen.setdefault("ids_torch", choice.topk(8, dim=-1).indices)
            seen.setdefault(
                "bias_stats",
                (bias.min().item(), bias.max().item(), bias.abs().mean().item(),
                 mod.gate.e_score_correction_bias.data_ptr()),
            )

    hook = L.module.mlp.register_forward_pre_hook(ffn_in)

    def routed(mod, args, out):
        """Record SGLang's routing, then route in f32 as the model defines it (the
        kernel does): SGLang's AITER top-k casts the correction bias (~34) to bf16."""
        seen.setdefault("ids", out.topk_ids.clone())
        seen.setdefault("weights", out.topk_weights.clone())
        sc = seen.get("scores_torch")
        if sc is None:
            return None
        bias = mod_gate.e_score_correction_bias.float()
        ids = (sc + bias).topk(8, dim=-1).indices
        raw = sc.gather(1, ids)
        w = raw / raw.sum(-1, keepdim=True) * 2.5
        routed_cols = out.topk_ids != 256
        new_ids, new_w = out.topk_ids.clone(), out.topk_weights.clone()
        new_ids[routed_cols] = ids.to(new_ids.dtype).reshape(-1)
        new_w[routed_cols] = w.to(new_w.dtype).reshape(-1)
        return out._replace(topk_weights=new_w, topk_ids=new_ids)

    mod_gate = L.module.mlp.gate

    hook_topk = L.module.mlp.topk.register_forward_hook(routed)
    hook_gate = L.module.mlp.gate.register_forward_hook(
        lambda mod, args, out: seen.setdefault(
            "scores", torch.sigmoid((out[0] if isinstance(out, tuple) else out).float())
        )
    )
    try:
        out, topk = L.module(
            positions[:S],
            hidden,
            forward_batch,
            zero,
            None,
            None,
            prev_topk_indices=prev_topk,
        )
    finally:
        hook.remove()
        hook_topk.remove()
        hook_gate.remove()
    ref = {
        "streams": take_output(fold(out, forward_batch), forward_batch).clone(),
        "topk": topk,
        "attn": seen.get("attn"),
        "ids": seen.get("ids"),
        "scores": seen.get("scores"),
        "ffn_in": seen.get("ffn_in"),
        "scores_torch": seen.get("scores_torch"),
        "ids_torch": seen.get("ids_torch"),
        "bias_stats": seen.get("bias_stats"),
        "weights": seen.get("weights"),
    }
    for name, (t, idx) in regions.items():
        ref[name] = t[idx].clone()
        t[idx] = saved[name]
    return ref


def _unshuffle(w: torch.Tensor) -> torch.Tensor:
    """AITER's (16, 16) FP8 layout [N / 16, K / 32, 2, 16, 16] back to row-major."""
    n, k = w.shape
    return w.view(n // 16, k // 32, 2, 16, 16).permute(0, 3, 1, 2, 4).reshape(n, k)


def mid_reference(L, x: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """SiLU(gate) * up of row 0 for each routing slot ``sel`` [slots], in f32 from
    SGLang's FP8 experts."""
    ex = L.module.mlp.experts
    outs = []
    for e in sel.tolist():
        w = ex.w13_weight[e]
        if L.w.shuffled:
            w = _unshuffle(w)
        sc = ex.w13_weight_scale_inv[e]
        wf = w.float() * sc.repeat_interleave(128, 0)[: w.shape[0]].repeat_interleave(
            128, 1
        )[:, : w.shape[1]]
        gu = wf @ x.float()
        inter = gu.shape[0] // 2
        outs.append(F.silu(gu[:inter]) * gu[inter:])
    return torch.stack(outs)


def read_mailbox(rt, name: str, n: int, dtype) -> torch.Tensor:
    """The first ``n`` values of a launch's (value, tag) mailbox."""
    off = rt.sc[name]
    words = rt.scratch[off : off + n * 8].view(torch.int32).view(-1, 2)[:, 0]
    return words.contiguous().view(dtype)


def read_bf16_mailbox(rt, name: str, n: int) -> torch.Tensor:
    """The first ``n`` values of a launch's bf16-pair mailbox (a word of two bf16,
    then its tag)."""
    off = rt.sc[name]
    words = rt.scratch[off : off + n // 2 * 8].view(torch.int32).view(-1, 2)[:, 0]
    return words.contiguous().view(torch.bfloat16)[:n]


def compare(
    L,
    ref: dict,
    streams: torch.Tensor,
    positions,
    forward_batch,
    S: int,
    attn=None,
    routing=None,
    scores=None,
    xqd=None,
    mid=None,
) -> None:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.glm53.spec import NUMERICS

    if get_parallel().tp_rank != 0:
        return
    outputs = _outputs("streams", streams, ref["streams"])
    if attn is not None and ref.get("attn") is not None:
        outputs.update(_outputs("attn", attn, ref["attn"][:S]))
    if scores is not None and ref.get("scores") is not None:
        outputs.update(_outputs("scores", scores, ref["scores"][:S]))
        if ref.get("scores_torch") is not None:
            outputs.update(_outputs("scores_t_vs_sgl", ref["scores_torch"][:S], ref["scores"][:S]))
            outputs.update(_outputs("scores_t_vs_kernel", ref["scores_torch"][:S], scores))
    if xqd is not None and ref.get("ffn_in") is not None:
        outputs.update(_outputs("ffn_in", xqd, ref["ffn_in"][:S]))
    if mid is not None:
        # validate the unshuffle on a projection SGLang runs itself (column parallel)
        lin = L.module.self_attn.q_b_proj
        x = torch.randn(1, lin.weight.shape[1], dtype=torch.bfloat16, device=lin.weight.device)
        y = lin(x)
        y = y[0] if isinstance(y, tuple) else y
        w = _unshuffle(lin.weight) if L.w.shuffled else lin.weight
        sc = lin.weight_scale_inv
        wf = w.float() * sc.repeat_interleave(128, 0)[: w.shape[0]].repeat_interleave(
            128, 1
        )[:, : w.shape[1]]
        outputs.update(_outputs("unshuffle_check", (x.float() @ wf.t()), y.float()))
    if mid is not None:
        # which layout SGLang left the experts in: against the checkpoint's expert 0
        import json
        import os

        from safetensors import safe_open
        from sglang.srt.runtime_context import process_model_config

        root = process_model_config().model_path
        name = f"model.layers.{L.i}.mlp.experts.0.gate_proj.weight"
        shard = json.load(open(os.path.join(root, "model.safetensors.index.json")))[
            "weight_map"
        ][name]
        with safe_open(os.path.join(root, shard), framework="pt") as f:
            gate = f.get_tensor(name)
        w13 = L.module.mlp.experts.w13_weight[0]
        inter = w13.shape[0] // 2
        want = gate[:inter].view(torch.uint8).to(w13.device)
        raw = w13[:inter].view(torch.uint8)
        un = _unshuffle(w13).view(torch.uint8)[:inter]
        logger.info(
            "monokernel check layer %d expert 0 gate rows 0..%d equal the checkpoint: "
            "as stored %s, unshuffled %s",
            L.i,
            inter,
            torch.equal(raw, want),
            torch.equal(un, want),
        )
    if mid is not None and routing is not None and ref.get("ffn_in") is not None:
        want = mid_reference(L, ref["ffn_in"][0], routing[0][0])
        outputs.update(_outputs("mid", mid[: want.shape[0]], want))
        logger.info(
            "monokernel check layer %d mid per slot (expert, |kernel|, |ref|, cos): %s",
            L.i,
            [
                (e, round(a.norm().item(), 3), round(b.norm().item(), 3),
                 round(F.cosine_similarity(a.float(), b.float(), dim=0).item(), 3))
                for e, a, b in zip(routing[0][0].tolist(), mid, want)
            ],
        )
    for name, (t, idx) in _regions(L, forward_batch, S).items():
        outputs.update(_outputs(name, t[idx], ref[name]))
    values = {
        f"layer.{out}.{metric}": v for out, ms in outputs.items() for metric, v in ms.items()
    }
    if ref.get("ids") is not None and L.i < 6:
        sel, prob = routing
        logger.info(
            "monokernel check layer %d torch top-8 %s, bias (min, max, mean |b|, ptr) %s, kernel bias ptr %s",
            L.i,
            sorted(ref["ids_torch"][0].tolist()) if ref.get("ids_torch") is not None else None,
            ref.get("bias_stats"),
            L.w.bias.data_ptr(),
        )
        logger.info(
            "monokernel check layer %d routing row 0: kernel %s | sglang %s",
            L.i,
            sorted(zip(sel[0].tolist(), [round(p, 4) for p in prob[0].tolist()])),
            sorted(
                zip(
                    ref["ids"][0].tolist(),
                    [round(w, 4) for w in ref["weights"][0].float().tolist()],
                )
            ),
        )
    pos = positions[:S]
    logger.info(
        "monokernel check layer %d (%s, %d rows, positions %d..%d): %s",
        L.i,
        "indexer" if L.w.index is not None else "shared",
        S,
        pos.min().item(),
        pos.max().item(),
        " ".join(f"{k}={v:.6g}" for k, v in values.items()),
    )
    for name, bound in NUMERICS.bounds.items():
        v = values.get(name)
        if v is None:
            continue
        ok = v >= bound if name.endswith(".cosine") else v <= bound
        if not ok:
            logger.error(
                "monokernel check violation: layer %d %s = %.6g, bound %g",
                L.i,
                name,
                v,
                bound,
            )
