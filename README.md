# Orthrus: Memory-Efficient Parallel Token Generation via Dual-View Diffusion

Official implementation and model checkpoints for **Orthrus**, a dual-architecture framework that unifies the exact generation fidelity of autoregressive Large Language Models (LLMs) with the high-speed parallel token generation of diffusion models.

<p align="center">
  <img src="assets/orthrus.png" width="80%" alt="Orthrus Architecture">
</p>

https://github.com/user-attachments/assets/2a0b021c-e232-4ac6-bf5c-c582c422505e

## Model Zoo
 
All models use a Qwen3 backbone and guarantee **strictly lossless generation**.
 
| Model | Base Model | HuggingFace | Avg. Speedup |
| :--- | :--- | :--- | :--- |
| Orthrus-Qwen3-1.7B | Qwen3-1.7B | [🤗 HuggingFace](https://huggingface.co/chiennv/Orthrus-Qwen3-1.7B) | 4.25× |
| Orthrus-Qwen3-4B | Qwen3-4.0B | [🤗 HuggingFace](https://huggingface.co/chiennv/Orthrus-Qwen3-4B) | 5.20× |
| Orthrus-Qwen3-8B | Qwen3-8.0B | [🤗 HuggingFace](https://huggingface.co/chiennv/Orthrus-Qwen3-8B) | 5.36× |
 
---
## Installation
 
```bash
uv pip install -e .
uv pip install ninja packaging
uv pip install flash-attn --no-build-isolation # or: pip install "flash-attn-4[cu13]" if your device supports it
```
 
> We recommend [`uv`](https://github.com/astral-sh/uv) for fast dependency resolution.

---
 
## Quickstart
 
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
 

model = AutoModelForCausalLM.from_pretrained(
    "chiennv/Orthrus-Qwen3-8B",
    dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="flash_attention_2",  # use flash_attention_4 if your system does support
    trust_remote_code=True,
).eval()
tokenizer = AutoTokenizer.from_pretrained("chiennv/Orthrus-Qwen3-8B")
 
prompt = "Write a program to count the frequency of each word in a paragraph."
messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False).input_ids

output_ids = model.generate(
    input_ids=input_ids.to(model.device), 
    max_new_tokens=2048,
    use_diffusion_mode=True, 
    streamer=TextStreamer(tokenizer, skip_prompt=True) # enable streaming generation
)
```

> **Coming soon:** Native integration with [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) is coming soon. Stay tuned!
 
## Key Advantages
 
- **Significant Inference Acceleration:** Breaks the sequential bottleneck of standard autoregressive decoding, delivering up to a $7.8\times$ speedup on generation tasks.
- **Strictly Lossless Generation:** Employs an exact intra-model consensus mechanism to guarantee that the output matches the original base model's exact predictive distribution.
- **Zero Redundant Memory Overhead:** Both the autoregressive and diffusion views attend to the exact same high-fidelity Key-Value (KV) cache natively, resulting in only an $O(1)$ memory cache overhead.
- **Parameter Efficient:** Parallel generation capabilities are injected by fine-tuning only 16% of the total model parameters while keeping the base LLM strictly frozen.

---

## Citation

If you find this model or architecture useful in your work, please cite our [paper](https://arxiv.org/abs/2605.12825):

```bibtex
@misc{vannguyen2026orthrusmemoryefficientparalleltoken,
      title={Orthrus: Memory-Efficient Parallel Token Generation via Dual-View Diffusion}, 
      author={Chien Van Nguyen and Chaitra Hegde and Van Cuong Pham and Ryan A. Rossi and Franck Dernoncourt and Thien Huu Nguyen},
      year={2026},
      eprint={2605.12825},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.12825}, 
}
```
