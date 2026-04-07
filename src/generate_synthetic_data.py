import json
import os
import time
import requests
from dotenv import load_dotenv
from src.security_layer import BudgetGuard, PromptShield
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ForgeSyntheticGen")

load_dotenv()

# We default to Groq as it is fast and previously used, but it can be swapped.
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
API_URL = "https://api.groq.com/openai/v1/chat/completions"

def generate_example(topic: str, guard: BudgetGuard) -> dict | None:
    """Generate a single AI/ML Q&A pair using an LLM, strictly adhering to the Reason/Answer structure."""
    
    # SECURITY: Scan input before processing
    PromptShield.scan(topic)
    
    prompt = (
        f"Generate a highly technical QA pair about {topic}. "
        "Format EXACTLY like this with no markdown code blocks:\n"
        "Question: [User question here]\n"
        "Reasoning:\n1. [step 1]\n2. [step 2]\n3. [step 3]\n\n"
        "Answer: [Concise answer here]"
    )
    
    system_instruction = "You are an expert AI/ML dataset generator. You only output strictly in the requested format."

    # Estimate token usage (very rough heuristic: 1 word ~ 1.3 tokens)
    estimated_tokens = len(prompt.split()) * 2 + 300 # prompt + expected output
    
    # SECURITY layer 4: Budget Check before API hitting
    guard.check_limit(estimated_tokens=estimated_tokens)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192", # Strong model for generation
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 512
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 429:
            logger.warning("Rate limit hit! Sleeping for 60 seconds...")
            time.sleep(60)
            return None # Skip and continue

        if response.status_code == 401:
            logger.error("Authentication Error (401). Check GROQ_API_KEY in .env.")
            return None

        response.raise_for_status()
        
        data = response.json()
        content = data['choices'][0]['message']['content'].strip()
        usage = data.get('usage', {})
        total_tokens = usage.get('total_tokens', estimated_tokens)
        
        # Success - Consume Budget
        guard.consume(step_increment=1, tokens=total_tokens)
        
        # Parse output cautiously
        if "Question:" in content and "Reasoning:" in content and "Answer:" in content:
            question_part = content.split("Question:")[1].split("Reasoning:")[0].strip()
            reasoning_answer_part = "Reasoning:" + content.split("Reasoning:")[1].strip()
            
            chatml_format = {
                "messages": [
                    {"role": "system", "content": "You are Forge, a precise AI/ML expert. Always answer with explicit Reasoning steps followed by a short Answer. Be concise and accurate."},
                    {"role": "user", "content": question_part},
                    {"role": "assistant", "content": reasoning_answer_part}
                ]
            }
            return chatml_format
        else:
            logger.warning(f"Failed to parse model output perfectly for topic: {topic}")
            return None

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout while generating topic {topic}")
        # Even if it times out, we consume a step to avoid infinite looping
        guard.consume(step_increment=1, tokens=estimated_tokens//2)
        return None
    except Exception as e:
        logger.error(f"API Error generating for {topic}: {e}")
        guard.consume(step_increment=1, tokens=estimated_tokens//2)
        return None

def main():
    if not GROQ_API_KEY:
        logger.error("No GROQ_API_KEY found in .env. Exiting generation.")
        return

    # A small targeted seed list to expand upon
    topics = [
        "Grouped Query Attention (GQA)", 
        "DPO vs PPO",
        "Sparse vs Dense Retrieval",
        "Rotary Position Embeddings (RoPE)",
        "KV Cache Offloading",
        "Mixture of Experts routing mechanisms",
        "Model distillation techniques"
    ]

    # Initialize Budget Guard: strict limit of 10 API calls, max 10k tokens
    budget_guard = BudgetGuard(max_steps=10, max_tokens_per_session=10000)
    
    # File to append to
    output_file = "data/train.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    generated_count = 0
    with open(output_file, 'a') as f:
        for idx, topic in enumerate(topics):
            logger.info(f"[{idx+1}/{len(topics)}] Generating sample for: {topic}")
            
            try:
                example = generate_example(topic, budget_guard)
                if example:
                    f.write(json.dumps(example) + "\n")
                    generated_count += 1
                
                # Respect rate limits (1 sec spacing)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Generation Loop Halted: {e}")
                break

    logger.info(f"Successfully generated and appended {generated_count} new examples to {output_file}.")
    logger.info(f"Budget Session ended. Steps used: {budget_guard.current_steps}/{budget_guard.max_steps}, Tokens used: {budget_guard.current_tokens}/{budget_guard.max_tokens}")

if __name__ == "__main__":
    main()
