# FORGE Security & Risk Mitigation

FORGE implements specialized security layers targeting common failure patterns in Agentic/Generative AI infrastructure: Infinite token burn loops, Prompt Injection, and Environment vulnerability.

## Layer 4 Budget Guard
A hard step limit is enforced at the execution layer (`src/security_layer.py`). When the system hits this limit—whether due to complex reasoning or an anomaly—it cleanly aborts.

**Behavior Checklist:**
- [x] Logs `CRITICAL: Agent terminated due to step limit (Budget Guard)`
- [x] Records termination state and contextual usage in the Query Ledger (`logs/query_ledger.json`)
- [x] Refuses further execution immediately
- [x] Avoids silent token bleeding API loops

## PromptShield Poisoning & Injection Defense
LLM applications are vulnerable to semantic manipulation where user inputs hijack system states.

**Mechanisms:**
- Scans user prompts before API consumption.
- Prevents special token impersonations (e.g., `<|im_start|>`, system override markers).
- Detects prompt poisoning intent (`forget previous instructions`, `you are now`).
- Prevents basic shell injection patterns during generation processes.

## Operational Risks

| Area | Risk | Mitigation |
|---|---|---|
| **API Keys** | HF_TOKEN or GROQ_API_KEY leaked in git history | `.env` in `.gitignore`, `.env.example` committed without values |
| **Model Weights** | Accidental upload of GB-sized merged models | `outputs/` in `.gitignore`, adapter-only recommendations |
| **Training Data** | Subpar or Malicious data polluting fine-tune | `generate_synthetic_data.py` relies on strictly validated prompt frameworks. |
| **Dependencies** | Supply chain attacks via PyPi packages | Strict dependency pinning in `requirements.txt` |
