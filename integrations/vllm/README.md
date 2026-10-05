# orthrus-vllm

A vLLM **out-of-tree plugin** that adds native `chiennv/Orthrus-*` checkpoint support to vLLM, with `trust_remote_code=False` — no vLLM fork, no monkeypatching, installed the same way any other vLLM model plugin is.

This exists because the equivalent in-tree vLLM PRs were closed (linked below). See [Why this is a plugin, not a vLLM PR](#why-this-is-a-plugin-not-a-vllm-pr) for the PR history and why an OOT plugin is vLLM's own documented path for this situation.

## Credit

This package integrates **Orthrus**, the original architecture and checkpoints, into vLLM. All of the model design, training, and released weights are the original authors' work:

> **Orthrus: Memory-Efficient Parallel Token Generation via Dual-View Diffusion**
> [chiennv2000/orthrus](https://github.com/chiennv2000/orthrus) · [arXiv:2605.12825](https://arxiv.org/abs/2605.12825) · Chien Nguyen
> Checkpoints: [`chiennv/Orthrus-Qwen3-1.7B`](https://huggingface.co/chiennv/Orthrus-Qwen3-1.7B), [`chiennv/Orthrus-Qwen3-4B`](https://huggingface.co/chiennv/Orthrus-Qwen3-4B), [`chiennv/Orthrus-Qwen3-8B`](https://huggingface.co/chiennv/Orthrus-Qwen3-8B)

This package contains no original model architecture or training contribution — it is an integration layer only.

## What this does

- Registers `OrthrusConfig` so `model_type="orthrus"` checkpoints load without remote code (`AutoConfig.register`).
- Registers `OrthrusForCausalLM` / `OrthrusLM` as a vLLM text-generation architecture (`ModelRegistry.register_model`).
- Serves Orthrus checkpoints through vLLM's **standard autoregressive path** — this is the fully working, supported part of this package.

## Diffusion mode (opt-in, experimental)

Diffusion-mode self-speculative decoding can be enabled with `ORTHRUS_VLLM_DIFFUSION=1` on vLLM 0.31.x. On the tested setup it gets about 52-55% draft acceptance but is currently **slower** than plain autoregressive decoding, and it is experimental. Output is identical to autoregressive in float32 (12 prompts x 96 tokens) but can differ in bfloat16 from numerical noise. It works by patching vLLM internals at plugin load, pinned to tested versions. See [`docs/DIFFUSION_MODE.md`](docs/DIFFUSION_MODE.md) for how to enable it, exactly what was validated, and what was not.

## Verification status

The autoregressive path was validated on real GPUs (A10G and A100-80GB) in the [#44792 PR thread](https://github.com/vllm-project/vllm/pull/44792), and re-run as this plugin on an A10G with vllm 0.31.0 (repo tests pass, greedy output on `chiennv/Orthrus-Qwen3-1.7B` matches the HF reference). Coverage is narrow: 1.7B only, short prompts. Before relying on it beyond experimentation, run the smoke test below yourself.

## Install

```bash
pip install vllm>=0.11.0
# from the repo root:
pip install -e integrations/vllm
# or, from inside this folder:
pip install -e .
```

The `register_orthrus` entry point loads automatically whenever vLLM starts, no explicit import needed. To load only this plugin among others, set `VLLM_PLUGINS=register_orthrus`.

## Usage

```python
from vllm import LLM, SamplingParams

llm = LLM(model="chiennv/Orthrus-Qwen3-1.7B", trust_remote_code=False)
outputs = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=32))
print(outputs[0].outputs[0].text)
```

## Why this is a plugin, not a vLLM PR

This started as two PRs directly against `vllm-project/vllm`:

- **[#44792](https://github.com/vllm-project/vllm/pull/44792)** — `[Model] Add Orthrus model support (autoregressive path)`.
- **[#53753](https://github.com/vllm-project/vllm/pull/53753)** — `[Model][Spec Decode] Orthrus diffusion-mode decoding (WIP)`, an explicit follow-up building on #44792.

Both were closed — see the linked threads for the full discussion — and this repository packages that same work as a standard OOT plugin instead, which doesn't require vllm-project/vllm's approval to ship.

## License

Apache-2.0 for the integration code in this repository (see [`LICENSE`](LICENSE)). The Orthrus architecture and checkpoints themselves are MIT-licensed by their original author — see [`NOTICE`](NOTICE) and the [original repository](https://github.com/chiennv2000/orthrus) for full attribution.
