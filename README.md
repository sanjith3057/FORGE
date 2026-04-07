# FORGE — Domain-Specific QLoRA Pipeline

FORGE is a QLoRA fine-tuning pipeline designed for low-VRAM (4GB RTX 2050) and Colab T4 hardware. It takes generic foundation models (like Llama-3.1-8B) and aligns them to answer AI/ML technical questions with a strict topological format: `Reasoning → Answer`.

## The Problem
Generic models approximate tasks. When asked for technical explanations, they often provide verbose, unstructured text. Prompt engineering fails to reliably enforce strict formatting under complex queries. Fine-tuning permanently aligns the model's behavior.

## Core Features
1. **Low VRAM Optimization**: Configured aggressively for local 4GB VRAM setups (RTX 2050). Uses double quantization, NF4, gradient check-pointing, and 8-bit AdamW.
2. **Layer 4 Budget Guard**: An absolute failure-prevention system enforcing maximum step execution and token burn per session, dropping infinite loops without token bleed.
3. **PromptShield**: Inference protection layer aggressively filtering out Jailbreaks, Token Injection, and Malicious instruction override.
4. **Before/After Analysis**: Real Streamlit UI explicitly built to compare 🧊 Base Model output against 🔥 Fine-Tuned structured predictability.

## Quick Start
1. `pip install -r requirements.txt`
2. Create `.env` from `.env.example`
3. Generate data: `python src/generate_synthetic_data.py` (Groq API required)
4. Train Model: `./scripts/run_training.sh`
5. Test UI (Before & After): `streamlit run ui/app.py`

## Live Comparison
The goal of FORGE is to demonstrate extreme behavioral shifts:

**Base Model Output:** *(Llama-3.1)*
> "The lost-in-the-middle problem is basically when models are given a lot of text and they just forget what was in the middle... It's a big problem in RAG."

**Fine-Tuned Output:** *(Forge-Llama-3.1)*
> **Reasoning:**
> 1. In long contexts, LLMs tend to pay more attention to the beginning and end.
> 2. Relevant chunks placed in the middle often get ignored.
> 
> **Answer:**
> The lost-in-the-middle problem occurs when LLMs fail to effectively use relevant information placed in the middle of long contexts.

## Security layer
Refer to `security.md` for in-depth insights into API budget management, and Prompt Shield configurations protecting against prompt poisoning.
