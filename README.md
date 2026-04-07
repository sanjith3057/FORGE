# 🔥 FORGE — QLoRA Fine-Tuning Pipeline

**Domain-Specific LLM Adaptation**

FORGE is a complete QLoRA fine-tuning pipeline that turns a general-purpose Llama-3.1-8B into a precise AI/ML expert. It teaches the model to answer technical questions with consistent structured output: **Reasoning** steps followed by a **concise Answer**.

No research lab. No H100 cluster. Just consumer hardware (or free Colab) and high-quality data.

[FORGE Dashboard]
<img width="1716" height="872" alt="Screenshot 2026-04-07 160456" src="https://github.com/user-attachments/assets/c78ef0ed-c780-4e97-9fb4-cdc68dafc9c2" />


---

## The Problem

Generic LLMs are trained to be good at everything — which means they are great at nothing specific.

Give Llama-3.1-8B a technical AI/ML question and it will:
- Sometimes use structured reasoning
- Often give verbose, unstructured answers
- Frequently miss consistent formatting

Prompt engineering can guide the model temporarily, but it doesn't change the underlying behavior.  
**Fine-tuning** changes the model weights so the desired behavior becomes the default.

Most developers assume fine-tuning requires expensive hardware and weeks of work.  
**FORGE proves you can do it in under 4 hours** on a consumer GPU with measurable results.

---
## Research

**LoRA: Low-Rank Adaptation — Hu et al., Microsoft, 2021**
Instead of updating all model weights during fine-tuning, LoRA freezes the base model and adds small trainable matrices to specific layers. The number of trainable parameters drops by up to 10,000x. The base model is untouched, which means you can switch adapters without reloading the full model. This is the foundational technique behind everything built on top of it.

**QLoRA: Efficient Fine-Tuning of Quantised LLMs — Dettmers et al., 2023**
QLoRA combines 4-bit quantisation of the base model with LoRA adapters. The base model runs in 4-bit precision — reducing memory by ~75% — while the adapter weights train in 16-bit. LoRA can reduce the number of trainable parameters by 10,000x, making fine-tuning Llama-3.1-8B with QLoRA practical on consumer hardware. This is what makes fine-tuning on an RTX 2050 viable.

**Preventing Overfitting and Hallucinations — NAACL 2025**
Preference fine-tuning using targeted synthetic datasets designed to be hard to hallucinate on produced 90–96% hallucination reductions while preserving overall quality. Data quality matters more than data quantity. 500 well-curated examples outperform 5,000 noisy ones every time.

**RAG vs Fine-Tuning — LaRA Benchmark, ICML 2025**
Put volatile knowledge in retrieval, put stable behaviour in fine-tuning, and stop trying to force one tool to do both jobs. Fine-tune when your failure mode is behaviour inconsistency — wrong format, unstable tone, weak classification, or poor policy adherence. If failures come from missing or stale facts, use RAG.

---
## Results

**Before vs After Comparison**

| Metric                        | Base Model (Llama-3.1)     | FORGE Fine-Tuned           | Improvement     |
|------------------------------|----------------------------|----------------------------|-----------------|
| Format Consistency           | 6 / 20 questions           | 19 / 20 questions          | +217%          |
| Structured Reasoning         | Inconsistent / Missing     | Present in every response  | Major          |
| Average Latency              | ~2.1s                      | ~1.4s                      | ~33% faster    |
| Response Style               | Verbose & generic          | Concise + Professional     | Clear          |

**Live Demo Video**

[Watch How FORGE Work](

https://github.com/user-attachments/assets/716a8a8b-8ac2-4435-86d8-571d44062835

)  

---

## How I approch the Building of Forge

**FORGE** is a full QLoRA fine-tuning pipeline focused on **AI/ML question answering**.

The fine-tuned model learns to always respond in this format:

```
Reasoning:
1. Step-by-step logical breakdown
2. ...

Answer: [Short, precise final answer]

Notes: [Optional]
```
It includes:
- Training script with Unsloth + QLoRA
- Professional Streamlit dashboard with live base vs tuned comparison
- Strong PromptShield to block prompt injection attacks
- Detailed analytics dashboard with charts and session summary
- Complete documentation

---

## The 4-Stage Pipeline

### Stage 1 — Dataset Preparation
Create 200–500 high-quality instruction-response pairs in **ChatML JSONL** format.  
Every assistant response must follow the strict structure:

```json
{
  "messages": [
    {"role": "system", "content": "You are a precise AI/ML expert..."},
    {"role": "user", "content": "What is the lost-in-the-middle problem in RAG?"},
    {"role": "assistant", "content": "Reasoning:\n1. ...\n\nAnswer: The lost-in-the-middle problem occurs when..."}
  ]
}
```

Quality matters more than quantity.

### Stage 2 — QLoRA Training
- Load Llama-3.1-8B in 4-bit NF4 quantization (75% memory reduction)
- Train only LoRA adapters (~0.1% of total parameters)
- Use Unsloth for 2× faster training and lower memory usage
- Works on RTX 2050 (4GB VRAM) with optimizations or free Colab T4

### Stage 3 — Evaluation
Run 20 held-out test questions through both base and fine-tuned models.  
Score on format consistency, reasoning presence, and accuracy.  
Generate before/after comparison table.

### Stage 4 — Professional Inference UI
Streamlit dashboard featuring:
- Side-by-side live comparison (Base vs FORGE)
- Rich analytics with multiple charts
- Detailed session summary report
- Real-time PromptShield protection

---

## Security Layer — PromptShield

FORGE includes a robust **PromptShield** that blocks common injection and jailbreak attempts in real-time, including:
- System prompt extraction
- DAN / "ignore previous instructions"
- Role confusion ("act as developer", "ADMIN MODE")
- Format bypass (base64, hex, etc.)
- Dangerous extraction requests

Blocked attempts are logged with timestamp and reason in the Security Log tab.

See [`security.md`](security.md) for full details.

---

## Quick Start (Inference Dashboard)

```bash
git clone https://github.com/sanjith3057/forge.git
cd forge

pip install -r requirements.txt

# Setup API key
cp ui/.env.example ui/.env
# Add your GROQ_API_KEY in ui/.env

streamlit run ui/app.py
```

---

## Research Foundation

- **LoRA** (Hu et al., Microsoft, 2021) — Low-Rank Adaptation
- **QLoRA** (Dettmers et al., 2023) — Quantized Low-Rank Adaptation for consumer hardware
- Data quality > quantity (NAACL 2025 findings)
- RAG vs Fine-tuning best practices (ICML 2025)

---

Other projects in the series:
- **PRISM-RAG** — Position-Aware Reranked Injected Sparse-dense Memory RAG
- **GUARDIAN-AGENT** — Self-healing ReAct agent
- **LENS** — Multimodal document intelligence

---

## License

MIT License — Free to use, modify, and learn from.

---

**Built by Sanjith G**  
---
