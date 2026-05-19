# Copyright © 2024 Apple Inc.

import random
import unittest
from typing import List

import mlx.core as mx

from mlx_lm.generate import (
    BatchGenerator,
    GenerationResponse,
    SequenceStateMachine,
    batch_generate,
    generate,
    generate_step,
    stream_generate,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load


class TestGenerate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        cls.model, cls.tokenizer = load(cls.HF_MODEL_PATH)
        cls.model.set_dtype(mx.float32)

    def test_generate(self):
        # Simple test that generation runs
        text = generate(
            self.model, self.tokenizer, "hello", max_tokens=5, verbose=False
        )

    def test_generate_with_logit_bias(self):
        logit_bias = {0: 2000.0, 1: -20.0}
        text = generate(
            self.model,
            self.tokenizer,
            "hello",
            max_tokens=5,
            logits_processors=make_logits_processors(logit_bias),
            verbose=False,
        )
        self.assertEqual(text, "!!!!!")

    def test_stream_generate_max_tokens(self):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a story about Einstein"}],
            tokenize=True,
            add_generation_prompt=True,
        )

        tokens = []
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=4,
        ):
            tokens.append(response.token)
        self.assertEqual(len(tokens), 4)

    def test_generate_with_processor(self):
        init_toks = self.tokenizer.encode("hello")

        all_toks = None

        def logits_processor(toks, logits):
            nonlocal all_toks
            all_toks = toks
            return logits

        generate(
            self.model,
            self.tokenizer,
            "hello",
            max_tokens=5,
            verbose=False,
            logits_processors=[logits_processor],
        )
        self.assertEqual(len(all_toks), len(init_toks) + 5)

    def test_stream_generate_speculative(self):
        # Use same model as draft model, this is not a speed test
        draft_model = self.model

        results: List[GenerationResponse] = []
        drafted: List[bool] = []

        # make a determinate sampler
        sampler = make_sampler(temp=0.0)
        messages = [{"role": "user", "content": "hello"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            draft_model=draft_model,
            num_draft_tokens=2,
            sampler=sampler,
        ):
            drafted.append(generation_result.from_draft)
            results.append(generation_result)

        self.assertEqual(len(results), 5)
        # since num_draft_tokens is 2 and draft model is the same, the
        # first 2 generations should be drafts, the third should come
        # from the target model, and last two should be drafts
        self.assertEqual(drafted, [True, True, False, True, True])

    def test_stream_generate_input_embeddings(self):
        sampler = make_sampler(temp=0.0)  # determinate sampler

        # get prompt embeddings
        messages = [{"role": "user", "content": "Say 'TEST' and nothing else"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        prompt_embeddings = self.model.model.embed_tokens(prompt)

        response = ""
        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            sampler=sampler,
            input_embeddings=prompt_embeddings,
        ):
            response += generation_result.text

        self.assertEqual("TEST", response)

    def test_stream_generate_input_embeddings_prefill(self):
        sampler = make_sampler(temp=0.0)  # determinate sampler

        # get prompt embeddings
        messages = [{"role": "user", "content": "Say 'TEST' and nothing else"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        prompt_embeddings = self.model.model.embed_tokens(prompt)

        # setup prompt progress callback to track batched prefill
        num_prompt_processing_callbacks = 0

        def progress_callback(processed: int, total: int) -> None:
            nonlocal num_prompt_processing_callbacks
            num_prompt_processing_callbacks += 1

        # generate
        prefill_step_size = 5
        response = ""
        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            sampler=sampler,
            input_embeddings=prompt_embeddings,
            prefill_step_size=prefill_step_size,
            prompt_progress_callback=progress_callback,
        ):
            response += generation_result.text

        self.assertEqual("TEST", response)
        num_embeddings = prompt_embeddings.shape[0]
        self.assertTrue(
            num_embeddings / prefill_step_size < num_prompt_processing_callbacks
        )

    def test_batch_matches_single(self):

        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model, stop_tokens=self.tokenizer.eos_token_ids, max_tokens=1
        )
        uids = gen.insert(prompts)
        batch_responses = {r.uid: r for r in gen.next_generated()}

        # Do a test for each prompt the logits are close
        for e, prompt in enumerate(prompts):

            for response in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=1
            ):
                blp = batch_responses[uids[e]].logprobs
                lp = response.logprobs
                self.assertTrue(mx.allclose(blp, lp))
                break

    def test_many_batches(self):

        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=1,
            prefill_batch_size=2,
            prefill_step_size=8,
            completion_batch_size=3,
        )
        uids = gen.insert(prompts)
        batch_responses = {}
        not_in = True
        iters = 0
        while responses := gen.next_generated():
            for r in responses:
                not_in &= r.uid not in batch_responses
                batch_responses[r.uid] = r
            iters += 1
        # only one token per prompt means only one response per prompt
        self.assertTrue(not_in)

        # completion batch size is too small for a single iteration
        self.assertTrue(iters > 1)

        # Do a test for each prompt the logits are close
        for e, prompt in enumerate(prompts):

            for response in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=1
            ):
                blp = batch_responses[uids[e]].logprobs
                lp = response.logprobs
                self.assertTrue(mx.allclose(blp, lp))
                break

    def test_batch_unique_max_toks(self):
        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            prefill_batch_size=2,
            prefill_step_size=8,
            completion_batch_size=3,
        )
        num_toks = [2, 3, 4, 5]
        uids = gen.insert(prompts, max_tokens=num_toks)
        batch_responses = {uid: [] for uid in uids}
        while responses := gen.next_generated():
            for r in responses:
                batch_responses[r.uid].append(r.token)

        # Do a test for each prompt the logits are close
        for e, prompt in enumerate(prompts):

            tokens = []
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=num_toks[e],
            ):
                tokens.append(response.token)

            batch_tokens = batch_responses[uids[e]]
            self.assertEqual(tokens, batch_tokens)

    def test_batch_sliding_window(self):
        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        self.model.make_cache = lambda: [
            RotatingKVCache(max_size=4) for _ in self.model.layers
        ]
        batch_gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=10,
            prefill_batch_size=1,
            prefill_step_size=8,
            completion_batch_size=2,
        )
        uids = batch_gen.insert(prompts)
        batch_responses = {uid: [] for uid in uids}
        while responses := batch_gen.next_generated():
            for r in responses:
                batch_responses[r.uid].append(r.logprobs)

        for e, uid in enumerate(uids):
            for i, response in enumerate(
                stream_generate(
                    self.model,
                    self.tokenizer,
                    prompts[e],
                    max_tokens=10,
                )
            ):
                batch_logprobs = batch_responses[uid][i]
                logprobs = response.logprobs
                self.assertTrue(
                    mx.allclose(batch_logprobs, logprobs, rtol=1e-4, atol=1e-4)
                )

        del self.model.make_cache

    def test_batch_generate_with_logits_processors(self):
        """Test that batch_generate with logits_processors produces correct results."""
        logit_bias = {0: 2000.0, 1: -2000.0}
        processors = make_logits_processors(logit_bias)

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=processors,
        )
        prompt = self.tokenizer.encode("hello")
        uids = batch_gen.insert([prompt])
        response = batch_gen.next_generated()[0]
        logprobs = response.logprobs
        self.assertEqual(logprobs[0].item(), 0.0)
        self.assertEqual(logprobs.argmin().item(), 1)

        del batch_gen

        logit_bias = {0: 2000.0}
        processors = make_logits_processors(logit_bias)
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=processors,
        )

        (uid0,) = batch_gen.insert([prompt])

        logit_bias = {1: 2000.0}
        processors = make_logits_processors(logit_bias)
        (uid1,) = batch_gen.insert([prompt], logits_processors=[processors])

        logit_bias = {2: 2000.0}
        processors = make_logits_processors(logit_bias)
        (uid2,) = batch_gen.insert([prompt], logits_processors=[processors])

        responses = batch_gen.next_generated()
        responses = {response.uid: response for response in responses}
        self.assertEqual(responses[uid0].logprobs[0].item(), 0.0)
        self.assertEqual(responses[uid1].logprobs[1].item(), 0.0)
        self.assertEqual(responses[uid2].logprobs[2].item(), 0.0)

    def test_batch_generate_processor_tokens_match_prompt_on_first_step(self):
        prompt = self.tokenizer.encode("hello")
        seen = []

        def processor(tokens, logits):
            seen.append(tokens)
            return logits

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=[processor],
        )
        batch_gen.insert([prompt])
        batch_gen.next_generated()

        self.assertTrue(hasattr(seen[0], "shape"))
        self.assertEqual(seen[0].tolist(), prompt)

    def test_batch_generate_function_with_logits_processors(self):
        """Test that batch_generate function with logits_processors produces correct results."""
        logit_bias = {0: 2000.0, 1: -2000.0}
        processors = make_logits_processors(logit_bias)

        prompts = [self.tokenizer.encode("hello")]
        response = batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            max_tokens=1,
            logits_processors=processors,
        )
        self.assertEqual(len(response.texts), 1)
        generated_token = self.tokenizer.encode(response.texts[0])[0]
        self.assertEqual(generated_token, 0)

    def test_batch_generate_with_samplers(self):
        """Test that batch_generate with logits_processors produces correct results."""
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            sampler=lambda _: mx.array([1]),
        )
        prompt = self.tokenizer.encode("hello")
        uids = batch_gen.insert([prompt])
        response = batch_gen.next_generated()[0]
        self.assertEqual(response.token, 1)

        del batch_gen

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            sampler=lambda _: mx.array([1]),
        )

        (uid0,) = batch_gen.insert([prompt])
        uid1, uid2 = batch_gen.insert(
            [prompt, prompt],
            samplers=[lambda _: mx.array([2]), lambda _: mx.array([3])],
        )

        responses = batch_gen.next_generated()
        responses = {response.uid: response for response in responses}
        self.assertEqual(responses[uid0].token, 1)
        self.assertEqual(responses[uid1].token, 2)
        self.assertEqual(responses[uid2].token, 3)

    def test_batch_generate_with_state_machines(self):
        """Test that batch_generate with per-sequence state_machines stops on different tokens."""
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=10,
        )
        prompt = self.tokenizer.encode("hello")

        sm_0 = SequenceStateMachine({"normal": [([0], None)]}, initial="normal")
        sm_1 = SequenceStateMachine({"normal": [([1], None)]}, initial="normal")
        sm_2 = SequenceStateMachine({"normal": [([2], None)]}, initial="normal")

        processor_0 = make_logits_processors({0: 2000.0})
        processor_1 = make_logits_processors({1: 2000.0})
        processor_2 = make_logits_processors({2: 2000.0})

        uid0, uid1, uid2 = batch_gen.insert(
            [prompt, prompt, prompt],
            logits_processors=[processor_0, processor_1, processor_2],
            state_machines=[sm_0, sm_1, sm_2],
        )

        responses = batch_gen.next_generated()
        responses = {response.uid: response for response in responses}

        self.assertEqual(responses[uid0].token, 0)
        self.assertEqual(responses[uid1].token, 1)
        self.assertEqual(responses[uid2].token, 2)
        self.assertEqual(responses[uid0].finish_reason, "stop")
        self.assertEqual(responses[uid1].finish_reason, "stop")
        self.assertEqual(responses[uid2].finish_reason, "stop")
        self.assertEqual(responses[uid0].match_sequence, (0,))
        self.assertEqual(responses[uid1].match_sequence, (1,))
        self.assertEqual(responses[uid2].match_sequence, (2,))

    def test_batch_continued_generation(self):
        for rotating in [False, True]:
            if rotating:
                self.model.make_cache = lambda: [
                    RotatingKVCache(max_size=4) for _ in self.model.layers
                ]

            # Make the prompts
            prompts_a = [
                "Write a story about Einstein",
                "Hi",
                "What time is it?",
                "How tall is Mt Everest?",
            ]
            prompts_a = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for p in prompts_a
            ]
            prompts_b = [
                "Another one",
                "sup?",
                "And how about the date?",
                "Mt Olympus?",
            ]
            prompts_b = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for p in prompts_b
            ]

            # Generate once
            batch_gen = BatchGenerator(
                self.model,
                stop_tokens=self.tokenizer.eos_token_ids,
                max_tokens=10,
                prefill_batch_size=4,
                prefill_step_size=8,
                completion_batch_size=2,
            )
            uids = batch_gen.insert(prompts_a)
            caches = {uid: None for uid in uids}
            while responses := batch_gen.next_generated():
                for r in responses:
                    if r.finish_reason is not None:
                        caches[r.uid] = r.prompt_cache
            caches = [caches[uid] for uid in uids]

            # Generate the 2nd time
            uids = batch_gen.insert(prompts_b, caches=caches)
            batch_responses = {uid: [] for uid in uids}
            while responses := batch_gen.next_generated():
                for r in responses:
                    batch_responses[r.uid].append(r.logprobs)

            for e, uid in enumerate(uids):
                for i, response in enumerate(
                    stream_generate(
                        self.model,
                        self.tokenizer,
                        prompts_b[e],
                        max_tokens=10,
                        prompt_cache=caches[e],
                    )
                ):
                    batch_logprobs = batch_responses[uid][i]
                    logprobs = response.logprobs
                    self.assertTrue(
                        mx.allclose(batch_logprobs, logprobs, rtol=1e-4, atol=1e-4)
                    )

            if rotating:
                del self.model.make_cache

    def _continued_generation_test_helper(self, model):
        def rand_prompt(n):
            return [random.randint(0, 1000) for _ in range(n)]

        # Make the prompts
        prompts_a = [
            rand_prompt(5),
            rand_prompt(3),
            rand_prompt(8),
            rand_prompt(1),
        ]
        prompts_b = [
            rand_prompt(2),
            rand_prompt(7),
            rand_prompt(4),
            rand_prompt(6),
        ]

        # Generate once
        batch_gen = BatchGenerator(
            model,
            stop_tokens={},
            max_tokens=10,
            prefill_batch_size=4,
            prefill_step_size=32,
            completion_batch_size=2,
        )

        uids = batch_gen.insert(prompts_a)
        caches = {uid: None for uid in uids}
        while responses := batch_gen.next_generated():
            for r in responses:
                if r.finish_reason is not None:
                    caches[r.uid] = r.prompt_cache

        caches = [caches[uid] for uid in uids]

        # Generate the 2nd time
        uids = batch_gen.insert(prompts_b, caches=caches)
        batch_responses = {uid: [] for uid in uids}
        while responses := batch_gen.next_generated():
            for r in responses:
                batch_responses[r.uid].append(r.logprobs)

        for e, uid in enumerate(uids):
            for i, (_, logprobs) in enumerate(
                generate_step(
                    mx.array(prompts_b[e]),
                    model,
                    max_tokens=10,
                    prompt_cache=caches[e],
                )
            ):
                batch_logprobs = batch_responses[uid][i]
                self.assertTrue(
                    mx.allclose(batch_logprobs, logprobs, rtol=1e-4, atol=1e-4)
                )

    def test_batch_continued_generation_ssm(self):
        from mlx_lm.models import mamba2

        random.seed(0)
        mx.random.seed(4)

        # Make a small SSM model
        args = mamba2.ModelArgs(
            model_type="mamba2",
            num_heads=8,
            head_dim=16,
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=128,
            state_size=32,
            num_hidden_layers=4,
            layer_norm_epsilon=1e-4,
            conv_kernel=3,
            n_groups=4,
            use_bias=False,
            use_conv_bias=False,
            tie_word_embeddings=True,
            time_step_limit=(0.01, 10),
            time_step_rank="auto",
        )
        model = mamba2.Model(args)
        self._continued_generation_test_helper(model)

    def test_batch_continued_generation_gated_delta(self):
        from mlx_lm.models import qwen3_next

        random.seed(0)
        mx.random.seed(4)
        args = qwen3_next.ModelArgs(
            model_type="qwen3_next",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=128,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1000,
            linear_num_value_heads=4,
            linear_num_key_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=3,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=128,
            mlp_only_layers=[0],
            moe_intermediate_size=128,
            rms_norm_eps=1e-5,
            head_dim=64,
            rope_theta=1000.0,
            partial_rotary_factor=0.5,
            max_position_embeddings=1000,
        )
        model = qwen3_next.Model(args)
        self._continued_generation_test_helper(model)

    def test_extend_cache_with_empty(self):
        from mlx_lm.generate import _extend_cache
        from mlx_lm.models.cache import make_prompt_cache

        cache_a = make_prompt_cache(self.model)

        prompt = mx.array([[1, 2, 3]])
        self.model(prompt, cache=cache_a)
        mx.eval([c.state for c in cache_a])

        result = _extend_cache(cache_a, [])
        self.assertEqual(len(result), len(cache_a))
        for c in result:
            self.assertGreater(c.offset, 0)

        result = _extend_cache([], cache_a)
        self.assertEqual(len(result), len(cache_a))
        for c in result:
            self.assertGreater(c.offset, 0)

    def test_remove_prompt_batch_updates_currently_processing(self):
        prompt_a = self.tokenizer.encode("Write a long story about a cat")
        prompt_b = self.tokenizer.encode("Write a long story about a dog")

        gen = BatchGenerator(
            self.model,
            max_tokens=5,
            prefill_batch_size=2,
            prefill_step_size=4,
            completion_batch_size=4,
        )
        uid_a, uid_b = gen.insert([prompt_a, prompt_b])

        gen.next()

        found = gen._find_uids([uid_a, uid_b])
        for uid in [uid_a, uid_b]:
            self.assertIn(uid, found)
            self.assertEqual(found[uid][0], 1)

        gen.remove([uid_a])

        self.assertEqual(len(gen._currently_processing), len(gen._prompt_batch))

        found = gen._find_uids([uid_b])
        self.assertIn(uid_b, found)

        while responses := gen.next_generated():
            if all(r.finish_reason is not None for r in responses):
                break

    def test_batch_max_kv_size_creates_rotating_cache(self):
        max_kv_size = 256
        gen = BatchGenerator(
            self.model,
            max_tokens=1,
            max_kv_size=max_kv_size,
        )

        prompt = self.tokenizer.encode("Write a long story about a cat")
        gen.insert([prompt])

        for r in gen.next_generated():
            if r.finish_reason is not None:
                for cache in r.prompt_cache:
                    self.assertIsInstance(cache, RotatingKVCache)
                    self.assertEqual(cache.max_size, max_kv_size)

    def test_batch_max_kv_size_limits_cache_growth(self):
        max_kv_size = 5
        gen = BatchGenerator(
            self.model,
            max_tokens=10,
            max_kv_size=max_kv_size,
            prefill_batch_size=1,
            prefill_step_size=128,
            completion_batch_size=1,
        )

        prompt = self.tokenizer.encode("Write a long story about a cat")
        gen.insert([prompt])

        for r in gen.next_generated():
            if r.finish_reason is not None:
                for cache in r.prompt_cache:
                    self.assertLessEqual(cache.keys.shape[2], max_kv_size)

    def test_batch_max_kv_size_none_creates_regular_cache(self):
        gen = BatchGenerator(
            self.model,
            max_tokens=1,
            max_kv_size=None,
        )

        prompt = self.tokenizer.encode("Write a long story about a cat")
        gen.insert([prompt])

        for r in gen.next_generated():
            if r.finish_reason is not None:
                for cache in r.prompt_cache:
                    self.assertIsInstance(cache, KVCache)

    def test_prefer_prefill_when_pending_default_false(self):
        # Default behavior must be unchanged: the new flag defaults to False.
        gen = BatchGenerator(self.model, max_tokens=1)
        self.assertFalse(gen._prefer_prefill_when_pending)

    def test_prefer_prefill_when_pending_accepted_and_stored(self):
        # Opting in stores the flag without affecting other init kwargs.
        gen = BatchGenerator(
            self.model,
            max_tokens=1,
            prefer_prefill_when_pending=True,
        )
        self.assertTrue(gen._prefer_prefill_when_pending)

    def test_prefer_prefill_pauses_decode_when_prefill_pending(self):
        # With the flag on, a step that has both queued prefill work and an
        # in-flight decode batch (below saturation) should skip the decode
        # this cycle and drain prefill first. With the flag off (default),
        # the decode runs as usual.
        prompt_a = self.tokenizer.encode("Write a long story about a cat")
        prompt_b = self.tokenizer.encode("Write a long story about a dog")

        def run(prefer_prefill):
            gen = BatchGenerator(
                self.model,
                max_tokens=5,
                prefill_batch_size=1,
                prefill_step_size=4,
                completion_batch_size=4,
                prefer_prefill_when_pending=prefer_prefill,
            )
            # Insert prompt A and drive next() until prefill has completed
            # and the sequence has been promoted into _generation_batch.
            # With prefill_step_size=4 and a longer prompt, the first sequence
            # needs multiple next() cycles to traverse prefill before it can
            # start decoding.
            gen.insert([prompt_a])
            for _ in range(20):
                gen.next()
                if len(gen._generation_batch) == 1:
                    break
            self.assertEqual(len(gen._generation_batch), 1)
            self.assertLess(len(gen._generation_batch), gen.completion_batch_size)

            # Queue a second prompt so prefill work is now pending while a
            # decode batch is in flight and not yet saturated.
            gen.insert([prompt_b])
            self.assertGreater(
                len(gen._unprocessed_sequences)
                + len(gen._currently_processing)
                + len(gen._prompt_batch),
                0,
            )

            tokens_before = gen._gen_tokens_counter
            gen.next()
            return gen._gen_tokens_counter - tokens_before

        decoded_with_flag_off = run(prefer_prefill=False)
        decoded_with_flag_on = run(prefer_prefill=True)

        # Flag off: the in-flight request gets a decode token this cycle.
        self.assertGreater(decoded_with_flag_off, 0)
        # Flag on: decode is paused so prefill can run first.
        self.assertEqual(decoded_with_flag_on, 0)


class TestPromptLookupDecoding(unittest.TestCase):
    """Tests for Prompt Lookup Decoding (PLD) and its bit-exact rollback.

    Mirrors the structure of TestGenerate: loads a small chat model once,
    then runs pure-Python tests for the n-gram lookup helper plus
    end-to-end tests against the model.
    """

    @classmethod
    def setUpClass(cls):
        cls.HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        cls.model, cls.tokenizer = load(cls.HF_MODEL_PATH)
        cls.model.set_dtype(mx.float32)

    def test_pld_find_draft_returns_match(self):
        from mlx_lm.generate import _pld_find_draft

        prompt = [10, 20, 30, 40, 50, 60, 70]
        generated = [99, 40, 50]
        # last 2 generated == [40, 50], match in prompt at idx 3,4 -> next 3 are [60, 70]
        out = _pld_find_draft(generated, prompt, k_lookback=2, k_lookahead=3)
        self.assertEqual(out, [60, 70])

    def test_pld_find_draft_no_match(self):
        from mlx_lm.generate import _pld_find_draft

        prompt = [10, 20, 30]
        generated = [100, 200, 300]
        out = _pld_find_draft(generated, prompt, k_lookback=2, k_lookahead=3)
        self.assertEqual(out, [])

    def test_pld_find_draft_picks_most_recent(self):
        from mlx_lm.generate import _pld_find_draft

        # Two matches; loop goes right-to-left and returns the most recent.
        prompt = [1, 2, 3, 99, 1, 2, 4, 5]
        generated = [9, 1, 2]
        out = _pld_find_draft(generated, prompt, k_lookback=2, k_lookahead=2)
        self.assertEqual(out, [4, 5])

    def test_pld_find_draft_insufficient_history(self):
        from mlx_lm.generate import _pld_find_draft

        prompt = [1, 2, 3, 4]
        generated = [1]  # k_lookback=2 > len(generated)
        out = _pld_find_draft(generated, prompt, k_lookback=2, k_lookahead=2)
        self.assertEqual(out, [])

    def test_prompt_lookup_generate_step_yield_shape(self):
        # Smoke test the step generator's output shape: each yield is
        # (int_token, mx.array_logprobs, bool_from_draft).
        from mlx_lm.generate import prompt_lookup_generate_step

        prompt = self.tokenizer.encode("hello world", return_tensors="mlx")[0]
        n = 0
        for tok, lp, from_draft in prompt_lookup_generate_step(
            prompt,
            self.model,
            prompt_lookup_num_tokens=3,
            prompt_lookup_min_match=2,
            max_tokens=4,
        ):
            self.assertIsInstance(tok, int)
            self.assertIsInstance(lp, mx.array)
            self.assertIsInstance(from_draft, bool)
            n += 1
        self.assertEqual(n, 4)

    def test_prompt_lookup_generate_step_matches_ar(self):
        # PLD with bit-exact rollback must yield the same tokens as plain
        # auto-regressive decoding under the same (greedy) sampler.
        from mlx_lm.generate import (
            generate_step,
            prompt_lookup_generate_step,
        )

        # A prompt with enough internal repetition to give PLD something
        # to draft from (otherwise it just falls back to AR every step).
        prompt = self.tokenizer.encode(
            "The cat sat on the mat. The cat sat on the mat. The cat sat on",
            return_tensors="mlx",
        )[0]

        ar_ids = []
        for i, (tok, _) in enumerate(generate_step(prompt, self.model)):
            ar_ids.append(tok)
            if i + 1 == 8:
                break

        pld_ids = []
        for tok, _, _ in prompt_lookup_generate_step(
            prompt,
            self.model,
            prompt_lookup_num_tokens=4,
            prompt_lookup_min_match=2,
            max_tokens=8,
        ):
            pld_ids.append(tok)

        self.assertEqual(pld_ids, ar_ids)

    def test_prompt_lookup_generate_step_rejects_bad_args(self):
        from mlx_lm.generate import prompt_lookup_generate_step

        prompt = self.tokenizer.encode("hi", return_tensors="mlx")[0]
        with self.assertRaises(ValueError):
            next(
                prompt_lookup_generate_step(
                    prompt, self.model, prompt_lookup_num_tokens=0, max_tokens=1
                )
            )
        with self.assertRaises(ValueError):
            next(
                prompt_lookup_generate_step(
                    prompt, self.model, prompt_lookup_min_match=0, max_tokens=1
                )
            )

    def test_stream_generate_prompt_lookup(self):
        # End-to-end wiring: stream_generate(prompt_lookup_num_tokens=...)
        # routes through prompt_lookup_generate_step and surfaces tokens.
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
        )
        n = 0
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=4,
            prompt_lookup_num_tokens=3,
        ):
            n += 1
        self.assertEqual(n, 4)

    def test_stream_generate_prompt_lookup_conflicts_with_draft(self):
        # draft_model and prompt_lookup_num_tokens are mutually exclusive.
        with self.assertRaises(ValueError):
            for _ in stream_generate(
                self.model,
                self.tokenizer,
                "hello",
                max_tokens=2,
                draft_model=self.model,
                prompt_lookup_num_tokens=2,
            ):
                pass
    def test_generate_step_snapkv_none_is_default(self):
        # Passing snapkv=None must not change the token sequence vs the
        # default (snapkv kwarg omitted entirely).
        prompt = mx.array(self.tokenizer.encode("hello world"))
        baseline = []
        for tok, _ in generate_step(
            prompt, self.model, max_tokens=8, sampler=lambda x: mx.argmax(x, axis=-1)
        ):
            baseline.append(int(tok.item()))
        with_kwarg = []
        for tok, _ in generate_step(
            prompt,
            self.model,
            max_tokens=8,
            sampler=lambda x: mx.argmax(x, axis=-1),
            snapkv=None,
        ):
            with_kwarg.append(int(tok.item()))
        self.assertEqual(baseline, with_kwarg)

    def test_generate_step_snapkv_min_ctx_gate_skips_short_prompt(self):
        # With min_ctx set above the prompt length, SnapKV must short-circuit
        # before touching the model and produce the same tokens as snapkv=None.
        prompt = mx.array(self.tokenizer.encode("hello world"))
        baseline = []
        for tok, _ in generate_step(
            prompt, self.model, max_tokens=8, sampler=lambda x: mx.argmax(x, axis=-1)
        ):
            baseline.append(int(tok.item()))
        gated = []
        for tok, _ in generate_step(
            prompt,
            self.model,
            max_tokens=8,
            sampler=lambda x: mx.argmax(x, axis=-1),
            snapkv={"min_ctx": 10**9},
        ):
            gated.append(int(tok.item()))
        self.assertEqual(baseline, gated)

    def test_generate_step_snapkv_skips_when_cache_already_populated(self):
        # _maybe_snapkv_prefill must not fire if the prompt cache is already
        # populated (mid-conversation reuse path). It detects that via
        # prompt_cache[0].offset > 0 and falls through to the standard path.
        from mlx_lm.models.cache import make_prompt_cache

        prompt = mx.array(self.tokenizer.encode("hello world"))

        # Pre-populate the cache by running one step manually.
        warm = make_prompt_cache(self.model)
        _ = self.model(prompt[None], cache=warm)
        mx.eval(_)
        self.assertGreater(warm[0].offset, 0)

        # min_ctx low enough to attempt SnapKV, but pre-populated cache
        # should force the fallback. Output should match a fresh-cache run
        # under the same warmed-cache continuation prompt.
        next_prompt = mx.array(self.tokenizer.encode(" how are you"))

        warm_copy = make_prompt_cache(self.model)
        _ = self.model(prompt[None], cache=warm_copy)
        mx.eval(_)

        baseline = []
        for tok, _ in generate_step(
            next_prompt,
            self.model,
            max_tokens=4,
            sampler=lambda x: mx.argmax(x, axis=-1),
            prompt_cache=warm_copy,
        ):
            baseline.append(int(tok.item()))

        with_snapkv = []
        for tok, _ in generate_step(
            next_prompt,
            self.model,
            max_tokens=4,
            sampler=lambda x: mx.argmax(x, axis=-1),
            prompt_cache=warm,
            snapkv={"min_ctx": 1},  # would fire but for the populated-cache gate
        ):
            with_snapkv.append(int(tok.item()))
        self.assertEqual(baseline, with_snapkv)


class TestAutoSpeculative(unittest.TestCase):
    """Tests for prompt-lookup decoding + auto-speculative router opt-in flags.

    These tests cover argument-parsing, helper-function correctness, and
    smoke import-paths. They do not exercise a full PLD/AR decode loop —
    that is covered indirectly by ``test_stream_generate_speculative`` and
    by the companion fork's regression suite.
    """

    @classmethod
    def setUpClass(cls):
        cls.HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        cls.model, cls.tokenizer = load(cls.HF_MODEL_PATH)
        cls.model.set_dtype(mx.float32)

    def test_module_exports_new_symbols(self):
        # New public symbols added by the auto-speculative router patch.
        from mlx_lm.generate import (
            _auto_spec_score,
            _pld_find_draft,
            auto_speculative_generate_step,
            prompt_lookup_generate_step,
        )

        self.assertTrue(callable(auto_speculative_generate_step))
        self.assertTrue(callable(prompt_lookup_generate_step))
        self.assertTrue(callable(_pld_find_draft))
        self.assertTrue(callable(_auto_spec_score))

    def test_setup_arg_parser_defaults(self):
        # Both flags default off: default behavior must be unchanged.
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        args = parser.parse_args(["--prompt", "hi"])
        self.assertFalse(args.auto_speculative)
        self.assertIsNone(args.prompt_lookup_num_tokens)

    def test_setup_arg_parser_auto_speculative_flag(self):
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        args = parser.parse_args(["--prompt", "hi", "--auto-speculative"])
        self.assertTrue(args.auto_speculative)
        self.assertIsNone(args.prompt_lookup_num_tokens)

    def test_setup_arg_parser_prompt_lookup_flag(self):
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        args = parser.parse_args(["--prompt", "hi", "--prompt-lookup-num-tokens", "5"])
        self.assertFalse(args.auto_speculative)
        self.assertEqual(args.prompt_lookup_num_tokens, 5)

    def test_pld_find_draft_basic_match(self):
        from mlx_lm.generate import _pld_find_draft

        # Prompt contains the bigram (4, 5); after a match on (4, 5) the
        # next two tokens (6, 7) should be returned as the draft.
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        generated = [9, 4, 5]
        draft = _pld_find_draft(generated, prompt, k_lookback=2, k_lookahead=2)
        self.assertEqual(draft, [6, 7])

    def test_pld_find_draft_no_match_returns_empty(self):
        from mlx_lm.generate import _pld_find_draft

        prompt = [1, 2, 3, 4, 5]
        generated = [98, 99]
        draft = _pld_find_draft(generated, prompt, k_lookback=2, k_lookahead=3)
        self.assertEqual(draft, [])

    def test_pld_find_draft_empty_inputs(self):
        from mlx_lm.generate import _pld_find_draft

        self.assertEqual(
            _pld_find_draft([], [1, 2, 3], k_lookback=2, k_lookahead=2), []
        )
        self.assertEqual(_pld_find_draft([1, 2], [], k_lookback=2, k_lookahead=2), [])

    def test_pld_find_draft_prefers_longer_match(self):
        from mlx_lm.generate import _pld_find_draft

        # Suffix (7, 8, 9) appears at one site; suffix (8, 9) appears
        # at two sites. Longer match wins.
        prompt = [1, 2, 3, 7, 8, 9, 100, 200, 8, 9, 300]
        generated = [5, 6, 7, 8, 9]
        draft = _pld_find_draft(generated, prompt, k_lookback=3, k_lookahead=1)
        self.assertEqual(draft, [100])

    def test_pld_find_draft_no_continuation(self):
        from mlx_lm.generate import _pld_find_draft

        # Match exists at the very end of the prompt: no follow-on tokens.
        prompt = [1, 2, 3, 4, 5]
        generated = [4, 5]
        draft = _pld_find_draft(generated, prompt, k_lookback=2, k_lookahead=3)
        self.assertEqual(draft, [])

    def test_auto_spec_score_short_prompt_is_zero(self):
        from mlx_lm.generate import _auto_spec_score

        score = _auto_spec_score(list(range(50)))
        self.assertEqual(score, 0.0)

    def test_auto_spec_score_long_prompt_is_positive(self):
        from mlx_lm.generate import _auto_spec_score

        # Long + highly repetitive prompt (10-token cycle * 200 = 2000
        # tokens; bigram density is essentially 1.0).
        repetitive = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 200
        score = _auto_spec_score(repetitive)
        self.assertGreater(score, 0.5)

    def test_auto_spec_score_in_unit_interval(self):
        from mlx_lm.generate import _auto_spec_score

        random.seed(0)
        for n in (300, 600, 1500, 5000):
            prompt = [random.randint(0, 1000) for _ in range(n)]
            score = _auto_spec_score(prompt)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_stream_generate_default_behavior_unchanged(self):
        # With neither auto_speculative nor prompt_lookup_num_tokens set,
        # output must match plain AR token-for-token under a determinate
        # sampler.
        sampler = make_sampler(temp=0.0)
        prompt = self.tokenizer.encode("hello")

        baseline = []
        for r in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=4,
            sampler=sampler,
        ):
            baseline.append(r.token)

        explicit_off = []
        for r in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=4,
            sampler=sampler,
            auto_speculative=False,
            prompt_lookup_num_tokens=None,
        ):
            explicit_off.append(r.token)

        self.assertEqual(baseline, explicit_off)

    def test_stream_generate_auto_speculative_short_prompt(self):
        # Short-prompt path of the router: must run cleanly and yield
        # max_tokens tokens (falls back to AR internally).
        sampler = make_sampler(temp=0.0)
        prompt = self.tokenizer.encode("hello")

        tokens = []
        for r in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=3,
            sampler=sampler,
            auto_speculative=True,
        ):
            tokens.append(r.token)

        self.assertEqual(len(tokens), 3)

    def test_stream_generate_auto_speculative_rejects_draft_model(self):
        # Auto-spec is mutually exclusive with draft_model.
        prompt = self.tokenizer.encode("hello")
        with self.assertRaises(ValueError):
            for _ in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=2,
                draft_model=self.model,
                auto_speculative=True,
            ):
                pass

    def test_stream_generate_prompt_lookup_rejects_draft_model(self):
        # PLD direct path is also mutually exclusive with draft_model.
        prompt = self.tokenizer.encode("hello")
        with self.assertRaises(ValueError):
            for _ in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=2,
                draft_model=self.model,
                prompt_lookup_num_tokens=4,
            ):
                pass


if __name__ == "__main__":
    unittest.main()
