# Copyright © 2026 Apple Inc.

"""Tests for the MTP speculative-decoding library.

Synthetic-only: no real model weights are loaded. Tests focus on shape /
contract behavior of the public API surface (``MTPHead``,
``load_mtp_head``, ``mtp_generate_step``, and the ``gated_delta_spec``
patch).
"""

import os
import random
import tempfile
import unittest

import mlx.core as mx
import mlx.utils

from mlx_lm.models import gated_delta_spec, qwen3_next
from mlx_lm.models.mtp_head import (
    MTPHead,
    MTPLayer,
    _strip_mtp_prefix,
    load_mtp_head,
    mtp_quant_predicate,
)
from mlx_lm.models.qwen3_5 import TextModelArgs
from mlx_lm.utils import quantize_model


def _tiny_text_args() -> TextModelArgs:
    """A small ``TextModelArgs`` that constructs an ``MTPHead`` cheaply.

    All sizes are kept tiny so the tests stay fast and memory-light. The
    fields here mirror the ``qwen3_next`` config used in
    ``test_batch_continued_generation_gated_delta``.
    """
    return TextModelArgs(
        model_type="qwen3_5_moe_text",
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        # MoE block fields.
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=64,
        moe_intermediate_size=64,
        # Rope params -- keep simple "default" type.
        rope_parameters={
            "type": "default",
            "rope_theta": 1000.0,
            "partial_rotary_factor": 0.5,
        },
    )


class TestMTPHead(unittest.TestCase):
    """``MTPHead`` constructs and exposes the expected sub-modules."""

    def test_mtp_head_constructs(self):
        args = _tiny_text_args()
        head = MTPHead(args)

        self.assertEqual(len(head.layers), 1)
        self.assertIsInstance(head.layers[0], MTPLayer)
        # Dual pre-FC norms + fc projection (2H -> H) + post norm.
        self.assertEqual(head.fc.weight.shape, (args.hidden_size, args.hidden_size * 2))
        self.assertEqual(head.pre_fc_norm_embedding.weight.shape, (args.hidden_size,))
        self.assertEqual(head.pre_fc_norm_hidden.weight.shape, (args.hidden_size,))
        self.assertEqual(head.norm.weight.shape, (args.hidden_size,))
        # MTPLayer is the full-attention branch (not the GDN branch).
        self.assertFalse(head.layers[0].is_linear)

    def test_mtp_head_forward_shape(self):
        args = _tiny_text_args()
        head = MTPHead(args)

        B, L, H = 1, 1, args.hidden_size
        hidden = mx.random.normal((B, L, H), dtype=mx.float32)
        token_emb = mx.random.normal((B, L, H), dtype=mx.float32)

        out = head(hidden, token_emb, cache=None)
        self.assertEqual(out.shape, (B, L, H))
        # Output should be finite (smoke check on the forward path).
        self.assertTrue(bool(mx.all(mx.isfinite(out)).item()))


class TestStripMTPPrefix(unittest.TestCase):
    """``_strip_mtp_prefix`` handles both ``language_model.mtp.*`` and
    ``mtp.*`` key layouts."""

    def test_strip_language_model_prefix(self):
        weights = {
            "language_model.mtp.fc.weight": mx.zeros((4, 8)),
            "language_model.mtp.norm.weight": mx.zeros((4,)),
            "language_model.model.layers.0.foo": mx.zeros((1,)),
        }
        out = _strip_mtp_prefix(weights)
        self.assertIn("fc.weight", out)
        self.assertIn("norm.weight", out)
        # Non-MTP keys are dropped.
        self.assertNotIn("language_model.model.layers.0.foo", out)
        self.assertEqual(len(out), 2)

    def test_strip_plain_mtp_prefix(self):
        weights = {
            "mtp.fc.weight": mx.zeros((4, 8)),
            "mtp.layers.0.input_layernorm.weight": mx.zeros((4,)),
        }
        out = _strip_mtp_prefix(weights)
        self.assertIn("fc.weight", out)
        self.assertIn("layers.0.input_layernorm.weight", out)
        self.assertEqual(len(out), 2)

    def test_strip_mixed_prefixes(self):
        # If both prefixes appear (a sidecar concatenated with a shard),
        # both should be stripped to the same key root.
        weights = {
            "language_model.mtp.norm.weight": mx.zeros((4,)),
            "mtp.fc.weight": mx.zeros((4, 8)),
            "model.embed_tokens.weight": mx.zeros((128, 4)),  # discarded
        }
        out = _strip_mtp_prefix(weights)
        self.assertIn("norm.weight", out)
        self.assertIn("fc.weight", out)
        self.assertEqual(len(out), 2)


class TestLoadMTPHead(unittest.TestCase):
    """``load_mtp_head`` reconstructs an ``MTPHead`` from a synthetic
    safetensors file."""

    # 8-bit/group_size=32 keeps the hidden_size=64 dims divisible.
    _QUANT_BITS = 8
    _QUANT_GROUP = 32

    def _write_synthetic_sidecar(self, prefix: str, args: TextModelArgs) -> str:
        """Construct + quantize an ``MTPHead`` and dump its weights under
        ``prefix`` to a temporary safetensors file. Quantizing first means
        the dumped key set (weight + scales + biases on every Linear)
        matches what ``load_mtp_head`` expects to receive.
        """
        head = MTPHead(args)
        head, _ = quantize_model(
            head,
            config={},
            group_size=self._QUANT_GROUP,
            bits=self._QUANT_BITS,
            mode="affine",
            quant_predicate=mtp_quant_predicate,
        )
        params = dict(mlx.utils.tree_flatten(head.parameters()))
        prefixed = {f"{prefix}{k}": v for k, v in params.items()}

        tmp_dir = tempfile.mkdtemp(prefix="mlx_lm_mtp_test_")
        path = os.path.join(tmp_dir, "model-mtp.safetensors")
        mx.save_safetensors(path, prefixed)
        return path

    def test_load_from_language_model_prefix(self):
        args = _tiny_text_args()
        path = self._write_synthetic_sidecar("language_model.mtp.", args)

        config = {"text_config": args.__dict__}
        head, loaded_args = load_mtp_head(
            path,
            config,
            group_size=self._QUANT_GROUP,
            bits=self._QUANT_BITS,
            mode="affine",
        )
        self.assertIsInstance(head, MTPHead)
        self.assertEqual(loaded_args.hidden_size, args.hidden_size)

    def test_load_from_plain_mtp_prefix(self):
        args = _tiny_text_args()
        path = self._write_synthetic_sidecar("mtp.", args)

        config = {"text_config": args.__dict__}
        head, _ = load_mtp_head(
            path,
            config,
            group_size=self._QUANT_GROUP,
            bits=self._QUANT_BITS,
            mode="affine",
        )
        self.assertIsInstance(head, MTPHead)

    def test_load_raises_on_no_mtp_keys(self):
        # A sidecar with no recognized MTP prefix should raise ValueError.
        tmp_dir = tempfile.mkdtemp(prefix="mlx_lm_mtp_test_")
        path = os.path.join(tmp_dir, "model-mtp.safetensors")
        mx.save_safetensors(path, {"unrelated.weight": mx.zeros((4, 4))})

        args = _tiny_text_args()
        config = {"text_config": args.__dict__}
        with self.assertRaises(ValueError):
            load_mtp_head(
                path,
                config,
                group_size=self._QUANT_GROUP,
                bits=self._QUANT_BITS,
                mode="affine",
            )


class TestGatedDeltaSpecPatch(unittest.TestCase):
    """``gated_delta_spec.install`` is idempotent and a no-op when no cache
    has ``_spec_capture`` set."""

    def test_install_is_idempotent(self):
        # Three calls in a row must not raise and must not double-wrap.
        gated_delta_spec.install()
        gated_delta_spec.install()
        gated_delta_spec.install()

        from mlx_lm.models.qwen3_5 import GatedDeltaNet as Qwen35GDN
        from mlx_lm.models.qwen3_next import Qwen3NextGatedDeltaNet

        self.assertTrue(getattr(Qwen35GDN, "_mlx_lm_spec_patched", False))
        self.assertTrue(getattr(Qwen3NextGatedDeltaNet, "_mlx_lm_spec_patched", False))

    def test_mark_and_unmark_capture(self):
        # Build a minimal cache-list-shaped object; the helpers must
        # tolerate (and skip) entries that are neither ArraysCache nor
        # KVCache.
        class _ArraysCacheStub:
            def __init__(self):
                self.cache = [mx.zeros((1,))]
                self.lengths = mx.array([0])
                self.left_padding = mx.array([0])

        class _KVCacheStub:
            offset = 5
            keys = mx.zeros((1, 1, 1, 1))

        a = _ArraysCacheStub()
        k = _KVCacheStub()
        gated_delta_spec.mark_for_capture([a, k])
        self.assertTrue(a._spec_capture)
        self.assertIsNone(a._spec_intermediate)
        # KV cache must not have a spurious _spec_capture attribute set.
        self.assertFalse(hasattr(k, "_spec_capture"))

        gated_delta_spec.unmark_capture([a, k])
        self.assertFalse(a._spec_capture)

    def test_rollback_to_intermediate_trims_kv_cache(self):
        class _KVCacheStub:
            def __init__(self, offset):
                self.offset = offset
                self.keys = mx.zeros((1, 1, 1, 1))

        k = _KVCacheStub(offset=7)
        # No ArraysCache in the list; rollback should still trim KV-style
        # caches and report ``True`` (no missing snapshots).
        ok = gated_delta_spec.rollback_to_intermediate([k])
        self.assertTrue(ok)
        self.assertEqual(k.offset, 6)


class TestMTPGenerateStep(unittest.TestCase):
    """``mtp_generate_step`` yields the documented ``(token, logprobs,
    from_draft)`` 3-tuple shape, end-to-end on a tiny synthetic model."""

    def test_yield_shape_and_token_count(self):
        random.seed(0)
        mx.random.seed(0)

        # Build a tiny qwen3_next model -- enough to exercise the
        # mtp_generate_step control flow.
        args = qwen3_next.ModelArgs(
            model_type="qwen3_next",
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
            linear_num_value_heads=4,
            linear_num_key_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=3,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=64,
            mlp_only_layers=[0],
            moe_intermediate_size=64,
            rms_norm_eps=1e-5,
            head_dim=16,
            rope_theta=1000.0,
            partial_rotary_factor=0.5,
            max_position_embeddings=256,
        )
        base = qwen3_next.Model(args)

        # mtp_generate_step expects a model wrapper that:
        #   - exposes ``language_model.model`` for the inner Qwen3-Next
        #     text-model hand-off (used by _full_forward_capture, etc.),
        #   - also forwards top-level attribute access (``layers``,
        #     ``make_cache``, ``lm_head``, etc.) so
        #     ``cache.make_prompt_cache(wrapper)`` and similar see the
        #     same surface a real ``Qwen3NextForCausalLM`` would expose.
        # The standalone ``qwen3_next.Model`` IS the text model wrapper
        # (it has ``.model``, ``.layers``, ``.lm_head`` itself), so the
        # test wrapper delegates everything else to it.
        class _Wrapper:
            def __init__(self, m):
                self.language_model = m

            def __getattr__(self, name):
                # Only invoked when normal attribute lookup fails.
                # Delegate to the inner model for ``layers``,
                # ``make_cache``, ``lm_head``, etc.
                return getattr(self.language_model, name)

        wrapper = _Wrapper(base)

        # Build a tiny MTPHead with matching dims via TextModelArgs.
        head = MTPHead(_tiny_text_args())

        # Local import to avoid cycles at module load.
        from mlx_lm.mtp_generate import mtp_generate_step

        prompt = mx.array([1, 2, 3, 4, 5], dtype=mx.uint32)
        max_tokens = 4
        results = []
        for tok, logprobs, from_draft in mtp_generate_step(
            prompt, wrapper, head, max_tokens=max_tokens
        ):
            results.append((tok, logprobs, from_draft))
            if len(results) >= max_tokens:
                break

        # Shape checks: each yield is (int, mx.array, bool).
        self.assertEqual(len(results), max_tokens)
        for tok, logprobs, from_draft in results:
            self.assertIsInstance(tok, int)
            self.assertTrue(0 <= tok < args.vocab_size)
            self.assertIsInstance(logprobs, mx.array)
            self.assertEqual(logprobs.shape, (args.vocab_size,))
            self.assertIsInstance(from_draft, bool)

        # The first yield is always the prefill-sampled token, never a draft.
        self.assertFalse(results[0][2])


if __name__ == "__main__":
    unittest.main()
