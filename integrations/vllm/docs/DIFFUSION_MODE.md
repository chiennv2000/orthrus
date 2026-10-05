# Diffusion-mode decoding (opt-in, experimental)

By default this plugin serves Orthrus through vLLM's standard autoregressive path. Diffusion-mode self-speculative decoding is available as an **opt-in, experimental** feature, pinned to the vLLM versions it was tested on (currently 0.31.x).

## Enable

```bash
export ORTHRUS_VLLM_DIFFUSION=1
```

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="chiennv/Orthrus-Qwen3-1.7B",
    trust_remote_code=False,
    speculative_config={
        "method": "dflash",  # vLLM's parallel-drafter slot, see below
        "model": "chiennv/Orthrus-Qwen3-1.7B",
        "num_speculative_tokens": 4,
    },
)
```

With the flag unset, nothing is patched and behaviour is the plain AR plugin. On an unsupported vLLM version, the flag raises instead of patching.

## How it works without an upstream change

vLLM has no registry for speculative drafters (`method="custom_class"` only hands the proposer token ids, with no attention metadata, KV cache or hidden states). But Orthrus' drafter has the same shape as a DFlash drafter: one bonus token plus K mask tokens in a single non-causal forward, reading the target's KV cache. So vLLM's `dflash` slot already does the config validation, aux-hidden-state plumbing and KV-group wiring it needs. `orthrus_vllm.speculative.activate` makes these changes when the flag is on:

- swaps `OrthrusProposer` in for `DFlashProposer` in `vllm.v1.worker.gpu_model_runner`
- registers the `DFlashOrthrusLM` architecture alias vLLM derives for the draft
- adds a `dflash_config` (mask token id, one aux layer) to `OrthrusConfig`
- sets `VLLM_USE_V2_MODEL_RUNNER=0`, because 0.31's V2 runner has its own speculator classes that are not patched

These are small, version-pinned patches, but they are still patches of vLLM internals and can break on any vLLM upgrade.

## What was validated

On a Modal A10G, vllm 0.31.0, `chiennv/Orthrus-Qwen3-1.7B`, `enforce_eager`, `num_speculative_tokens=4`, greedy (the float32/bfloat16 comparison used 12 prompts x 96 tokens; the acceptance and timing numbers come from 3-prompt x 64-token runs):

- **float32: output is identical to plain autoregressive decoding** on 12 prompts x 96 tokens (1,152 tokens, no divergence). This is the check that the proposer and verification logic are correct.
- **bfloat16: output is not always identical.** On the same 12 prompts only 5 matched, diverging at scattered positions (tokens 7 to 85). Verification runs the target with different query shapes than plain decoding, so kernels round differently and near-tied greedy tokens can flip. Because float32 matches exactly, this looks like numerical noise and not a logic bug, but the margin of the flipped tokens was not inspected.
- About 52-55% of draft tokens accepted (181/332 and 175/348 in two 3-prompt runs).
- It is **slower** than autoregressive here (about 2.6 s vs 1.1 s for the same batch, 0.44x).

## Not validated

- Bit-exact equality with autoregressive decoding in bfloat16. It is lossless in float32 only, in the tests run.
- Any speedup. The earlier upstream attempt also measured a slowdown, and the reference implementation's own 2.39x is not reproduced through vLLM. Likely contributors are eager mode, tiny batches, and the proposer's extra per-step work, but this was not profiled.
- CUDA graphs, tensor parallelism, large batches, long outputs, and the 4B/8B checkpoints.
- vLLM versions other than 0.31.x, and the V2 model runner.
