# `benjamin-levin/mlx-lm` — fork of [`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm)

> **Interim fork**: carries seven in-flight upstream draft PRs assembled into `main` (plus one on its own branch) so users can install the full Python-level optimization stack as one piece while the PRs land upstream individually. Will be retired once each PR merges into `ml-explore/mlx-lm`.
>
> `setup.py` pins `mlx` to the matching [`benjamin-levin/mlx` fork](https://github.com/benjamin-levin/mlx) so the C++/Metal-level optimizations stack with the Python-level ones. Building requires Xcode CLT + CMake.

## What's different from upstream

Measured impact (M4 Max 36 GB, Qwen3.6-35B-A3B-4bit unless noted; per-PR isolated re-measurement lives in each draft PR's body):

| Feature | Effect | Draft PR |
|---|---|---|
| **`BatchGenerator(prefer_prefill_when_pending=True)`** — opt-in scheduling fix: pauses decode while any prefill chunk is queued, so N>1 batches decode together at native batched speed instead of starving while one request prefills. Default off. | **2.58× @ N=3 ctx=8k, 1.71× @ N=3 ctx=32k**, ±0.4% (no-op) at N=1. | [#1](https://github.com/benjamin-levin/mlx-lm/pull/1) |
| **Opt-in bf16 GDN state** (`MLX_LM_GDN_STATE_BF16=1`) — allocates the Qwen3-Next gated-delta recurrent state as bf16 instead of fp32 to halve per-step state bandwidth. Teacher-forced KL ≤ 0.0068, top-1 match 95-96/96. | Kernel −8.9%; e2e modest in this fork's vanilla baseline, larger when stacked with the rest. | [#2](https://github.com/benjamin-levin/mlx-lm/pull/2) |
| **Qwen3-Next MoE compile-block** — wraps the routing and post-expert blend math in `@mx.compile` helpers so each MoE layer dispatches fewer graph entries per token. Heavy `switch_mlp` / `shared_expert` calls untouched. | Bit-exact; +0.4–1.1% e2e in isolation (within noise on this baseline — kept as structural cleanup). | [#3](https://github.com/benjamin-levin/mlx-lm/pull/3) |
| **SnapKV long-context KV compression** — opt-in `snapkv=dict(...)` kwarg on `generate_step` plus a new `SnapKVCache`. Captures last-window queries to score positions, keeps sink + top-k + recent window, drops the rest. | **1.21× @ 32k, 1.68× @ 95k, 1.94× @ 128k.** Clean no-op below 49k length gate. | [#4](https://github.com/benjamin-levin/mlx-lm/pull/4) |
| **Depth-1 MTP speculative decoding** — library API (`mtp_generate_step`) reusing the MTP head shipped with upstream Qwen3-Next / Qwen3.5-MoE checkpoints. Bit-exact under greedy decoding. | **1.13–1.21×** across echo / code / open-gen / qa-short workloads. Depth-2+ confirmed worse on M-series (always slower than d=1). | [#5](https://github.com/benjamin-levin/mlx-lm/pull/5) |
| **Persistent disk-backed prompt cache** — opt-in `disk_cache_dir` kwarg on `LRUPromptCache` + matching CLI flags on `mlx_lm.server`. Atomic safetensors writes, LRU byte-budget eviction. | **1477–4585× TTFT speedup** on cached prefixes (4k–96k), bit-exact. | [#6](https://github.com/benjamin-levin/mlx-lm/pull/6) |
| **Generic cache `snap()` / `restore()` + bit-exact PLD generator** — adds `snap()`/`restore()` to every built-in cache class plus a new `prompt_lookup_generate_step` using snapshot+restore for bit-exact rollback. Unlocks PLD on non-trimmable caches (GDN, Mamba `ArraysCache`) that the existing trim-based PLD doesn't support. | Bit-exact (10/10 prompts vs AR). **1.40–1.68× on GDN ArraysCache echo/code-edit workloads** — previously unreachable. Snap/restore overhead 0.04–0.14 μs per cache class. | [#8](https://github.com/benjamin-levin/mlx-lm/pull/8) |
| *(Not in `main`)* **Auto-speculative router** — opt-in `auto_speculative=True` kwarg on `generate_step` that routes between PLD and plain AR based on prompt length, n-gram density, and a 16-token PLD probe. | Conflicts with the bit-exact PLD chosen for `main` (both define `prompt_lookup_generate_step`). Available on the `auto-speculative-router` branch — `git checkout auto-speculative-router` if you want it. | [#7](https://github.com/benjamin-levin/mlx-lm/pull/7) |

### Why fusions and rollback primitives dominate this stack

The M4 series and below have **no native int4 / int8 GPU matmul instructions** — M5's Neural Accelerators added these, but anything M4 or earlier dequantizes every quantized weight to bf16 in registers before the matmul fires, on every use. KV-quant in particular only pays off when the dequant is fused into the attention kernel (the stock 3-launch quantized-SDPA path is *slower* than bf16 SDPA at long context). Most of the fusions in the matching [mlx fork](https://github.com/benjamin-levin/mlx) and the speculative-decoding paths here exist to amortize that dequant cost, or to bypass it entirely by reducing the number of times each weight is touched per generated token.

## Install

```bash
pip install git+https://github.com/benjamin-levin/mlx-lm.git@main
```

This installs `mlx-lm` from the fork and auto-pulls the matching [mlx fork](https://github.com/benjamin-levin/mlx) via `setup.py`. Build takes ~5–10 min on M-series (Xcode CLT + CMake required).

## Context

These changes were extracted from an [optimization study](https://github.com/benjamin-levin/mlx-fast) of 28 strategies attempted on Qwen3.6-35B-A3B-4bit on M4 Max (13 shipped, 15 documented as dead ends, with per-strategy methodology + measurement). Each draft PR on this fork has the full per-feature methodology, in-PR re-measurement, and any honest corrections vs originally-claimed wins.

---

## MLX LM 

MLX LM is a Python package for generating text and fine-tuning large language
models on Apple silicon with MLX.

Some key features include:

* Integration with the Hugging Face Hub to easily use thousands of LLMs with a
  single command. 
* Support for quantizing and uploading models to the Hugging Face Hub.
* [Low-rank and full model
  fine-tuning](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
  with support for quantized models.
* Distributed inference and fine-tuning with `mx.distributed`

The easiest way to get started is to install the `mlx-lm` package:

**With `pip`**:

```sh
pip install mlx-lm
```

**With `conda`**:

```sh
conda install -c conda-forge mlx-lm
```

### Quick Start

To generate text with an LLM use:

```bash
mlx_lm.generate --prompt "How tall is Mt Everest?"
```

To chat with an LLM use:

```bash
mlx_lm.chat
```

This will give you a chat REPL that you can use to interact with the LLM. The
chat context is preserved during the lifetime of the REPL.

Commands in `mlx-lm` typically take command line options which let you specify
the model, sampling parameters, and more. Use `-h` to see a list of available
options for a command, e.g.:

```bash
mlx_lm.generate -h
```

The default model for generation and chat is
`mlx-community/Llama-3.2-3B-Instruct-4bit`.  You can specify any MLX-compatible
model with the `--model` flag. Thousands are available in the
[MLX Community](https://huggingface.co/mlx-community) Hugging Face
organization.

### Python API

You can use `mlx-lm` as a module:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)
```

To see a description of all the arguments you can do:

```
>>> help(generate)
```

Check out the [generation
example](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/examples/generate_response.py)
to see how to use the API in more detail. Check out the [batch generation
example](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/examples/batch_generate_response.py)
to see how to efficiently generate continuations for a batch of prompts.

The `mlx-lm` package also comes with functionality to quantize and optionally
upload models to the Hugging Face Hub.

You can convert models using the Python API:

```python
from mlx_lm import convert

repo = "mistralai/Mistral-7B-Instruct-v0.3"
upload_repo = "mlx-community/My-Mistral-7B-Instruct-v0.3-4bit"

convert(repo, quantize=True, upload_repo=upload_repo)
```

This will generate a 4-bit quantized Mistral 7B and upload it to the repo
`mlx-community/My-Mistral-7B-Instruct-v0.3-4bit`. It will also save the
converted model in the path `mlx_model` by default.

To see a description of all the arguments you can do:

```
>>> help(convert)
```

#### Streaming

For streaming generation, use the `stream_generate` function. This yields
a generation response object.

For example,

```python
from mlx_lm import load, stream_generate

repo = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
model, tokenizer = load(repo)

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(response.text, end="", flush=True)
print()
```

#### Sampling

The `generate` and `stream_generate` functions accept `sampler` and
`logits_processors` keyword arguments. A sampler is any callable which accepts
a possibly batched logits array and returns an array of sampled tokens.  The
`logits_processors` must be a list of callables which take the token history
and current logits as input and return the processed logits. The logits
processors are applied in order.

Some standard sampling functions and logits processors are provided in
`mlx_lm.sample_utils`.

### Command Line

You can also use `mlx-lm` from the command line with:

```
mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.3 --prompt "hello"
```

This will download a Mistral 7B model from the Hugging Face Hub and generate
text using the given prompt.

For a full list of options run:

```
mlx_lm.generate --help
```

To quantize a model from the command line run:

```
mlx_lm.convert --model mistralai/Mistral-7B-Instruct-v0.3 -q
```

For more options run:

```
mlx_lm.convert --help
```

You can upload new models to Hugging Face by specifying `--upload-repo` to
`convert`. For example, to upload a quantized Mistral-7B model to the
[MLX Hugging Face community](https://huggingface.co/mlx-community) you can do:

```
mlx_lm.convert \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    -q \
    --upload-repo mlx-community/my-4bit-mistral
```

Models can also be converted and quantized directly in the
[mlx-my-repo](https://huggingface.co/spaces/mlx-community/mlx-my-repo) Hugging
Face Space.

### Long Prompts and Generations 

`mlx-lm` has some tools to scale efficiently to long prompts and generations:

- A rotating fixed-size key-value cache.
- Prompt caching

To use the rotating key-value cache pass the argument `--max-kv-size n` where
`n` can be any integer. Smaller values like `512` will use very little RAM but
result in worse quality. Larger values like `4096` or higher will use more RAM
but have better quality.

Caching prompts can substantially speedup reusing the same long context with
different queries. To cache a prompt use `mlx_lm.cache_prompt`. For example:

```bash
cat prompt.txt | mlx_lm.cache_prompt \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --prompt - \
  --prompt-cache-file mistral_prompt.safetensors
``` 

Then use the cached prompt with `mlx_lm.generate`:

```
mlx_lm.generate \
    --prompt-cache-file mistral_prompt.safetensors \
    --prompt "\nSummarize the above text."
```

The cached prompt is treated as a prefix to the supplied prompt. Also notice
when using a cached prompt, the model to use is read from the cache and need
not be supplied explicitly.

Prompt caching can also be used in the Python API in order to avoid
recomputing the prompt. This is useful in multi-turn dialogues or across
requests that use the same context. See the
[example](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/chat.py)
for more usage details.

### Supported Models

`mlx-lm` supports thousands of LLMs available on the Hugging Face Hub. If the
model you want to run is not supported, file an
[issue](https://github.com/ml-explore/mlx-lm/issues/new) or better yet, submit
a pull request. Many supported models are available in various quantization
formats in the [MLX Community](https://huggingface.co/mlx-community) Hugging
Face organization.

For some models the tokenizer may require you to enable the `trust_remote_code`
option. You can do this by passing `--trust-remote-code` in the command line.
If you don't specify the flag explicitly, you will be prompted to trust remote
code in the terminal when running the model. 

Tokenizer options can also be set in the Python API. For example:

```python
model, tokenizer = load(
    "qwen/Qwen-7B",
    tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
)
```

### Large Models

> [!NOTE]
    This requires macOS 15.0 or higher to work.

Models which are large relative to the total RAM available on the machine can
be slow. `mlx-lm` will attempt to make them faster by wiring the memory
occupied by the model and cache. This requires macOS 15 or higher to
work.

If you see the following warning message:

> [WARNING] Generating with a model that requires ...

then the model will likely be slow on the given machine. If the model fits in
RAM then it can often be sped up by increasing the system wired memory limit.
To increase the limit, set the following `sysctl`:

```bash
sudo sysctl iogpu.wired_limit_mb=N
```

The value `N` should be larger than the size of the model in megabytes but
smaller than the memory size of the machine.
