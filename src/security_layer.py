import logging
import re
import os
import json
from datetime import datetime
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ForgeSecurity")

class BudgetGuard:
    """
    Layer 4 - Budget Guard and Loop Detection.
    Enforces a hard step limit on LLM API calls to prevent infinite loops and token bleed.
    No silent failure. Logs critically on termination.
    """
    def __init__(self, max_steps: int = 10, max_tokens_per_session: int = 4000):
        self.max_steps = max_steps
        self.max_tokens = max_tokens_per_session
        self.current_steps = 0
        self.current_tokens = 0
        self.query_ledger = "logs/query_ledger.json"
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.query_ledger), exist_ok=True)

    def _record_ledger(self, reason: str, context: str):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason,
            "steps_consumed": self.current_steps,
            "tokens_consumed": self.current_tokens,
            "context": context
        }
        
        # Append to Ledger
        try:
            ledger_data = []
            if os.path.exists(self.query_ledger):
                with open(self.query_ledger, "r") as f:
                    ledger_data = json.load(f)
            ledger_data.append(record)
            with open(self.query_ledger, "w") as f:
                json.dump(ledger_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write to Query Ledger: {e}")

    def check_limit(self, estimated_tokens: int = 0) -> bool:
        """
        Call this before every LLM interaction.
        Returns empty tuple normally. If budget exceeded, raises Exception.
        """
        if self.current_steps >= self.max_steps:
            msg = "CRITICAL: Agent terminated due to step limit (Budget Guard)"
            logger.critical(msg)
            self._record_ledger("Step Limit Reached", "Loop detected or max step iteration hit.")
            raise Exception(msg)
            
        if self.current_tokens + estimated_tokens > self.max_tokens:
            msg = "CRITICAL: Agent terminated due to token budget exceeded (Budget Guard)"
            logger.critical(msg)
            self._record_ledger("Token Limit Reached", f"Attempted to consume more than {self.max_tokens} tokens.")
            raise Exception(msg)

        return True

    def consume(self, step_increment: int = 1, tokens: int = 0):
        """Record usage after a successful call."""
        self.current_steps += step_increment
        self.current_tokens += tokens


class PromptShield:
    """
    Hardened Prompt Injection & Jailbreak Detection.
    Covers:
    - Persona hijacking ("you are now", "act as", "pretend you are")
    - Safety bypass ("no safety filters", "no restrictions", "ignore guidelines")
    - DAN / jailbreak templates
    - Special token injection (<|im_start|>, ###, [INST])
    - Dangerous knowledge extraction (model inversion, training data extraction)
    - Command injection (bash, SQL, system calls)
    - Role override attacks
    """

    # ---- INPUT PATTERNS (block before sending to model) ----
    INPUT_PATTERNS = [
        # Classic instruction overrides
        re.compile(r"ignore (all |previous |prior |your |the )?instructions?", re.IGNORECASE),
        re.compile(r"disregard (all |previous |your |the )?instructions?", re.IGNORECASE),
        re.compile(r"forget (everything|what you|your instructions|your training)", re.IGNORECASE),
        re.compile(r"do not follow", re.IGNORECASE),
        re.compile(r"override (your |all )?(previous |prior )?(instructions?|guidelines?|training)", re.IGNORECASE),

        # Persona hijacking / role switching
        re.compile(r"you are (now |a |an )?(?!forge|an ai|a precise).*?(assistant|bot|ai|model|entity|system)", re.IGNORECASE),
        re.compile(r"act as (a |an )?(?!forge)", re.IGNORECASE),
        re.compile(r"pretend (you are|to be|that you)", re.IGNORECASE),
        re.compile(r"roleplay as", re.IGNORECASE),
        re.compile(r"your (new |true )?(role|persona|identity|name) is", re.IGNORECASE),
        re.compile(r"from now on (you are|act as|behave as|respond as)", re.IGNORECASE),
        re.compile(r"(switch|change) (to |your )?(a |an )?(different )?(mode|persona|role|identity)", re.IGNORECASE),

        # Safety bypass attempts
        re.compile(r"no (safety|content|ethical|moral|output)? ?(filters?|restrictions?|limitations?|guardrails?|guidelines?)", re.IGNORECASE),
        re.compile(r"without (any )?(restrictions?|limitations?|filters?|censorship)", re.IGNORECASE),
        re.compile(r"(ignore|bypass|disable|remove) (your )?(safety|content|ethical) (filter|guard|check|policy)", re.IGNORECASE),
        re.compile(r"you have no restrictions", re.IGNORECASE),
        re.compile(r"(jailbreak|DAN|do anything now)", re.IGNORECASE),
        re.compile(r"developer mode", re.IGNORECASE),
        re.compile(r"(unrestricted|uncensored|unfiltered) (mode|version|ai|assistant)", re.IGNORECASE),

        # System prompt leaking
        re.compile(r"(reveal|show|print|output|repeat|tell me) (your |the )?(system|initial|original|hidden)? ?(prompt|instruction|context|config)", re.IGNORECASE),
        re.compile(r"what (is|are|was) (your|the) (system )?prompt", re.IGNORECASE),

        # Special token injection
        re.compile(r"<\|im_start\|>", re.IGNORECASE),
        re.compile(r"<\|im_end\|>", re.IGNORECASE),
        re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE),
        re.compile(r"<\|system\|>|<\|user\|>|<\|assistant\|>", re.IGNORECASE),
        re.compile(r"###\s*(system|instruction|human|assistant)", re.IGNORECASE),

        # Dangerous knowledge extraction
        re.compile(r"(extract|leak|steal|exfiltrate) (training )?(data|weights|parameters|gradients)", re.IGNORECASE),
        re.compile(r"model (inversion|extraction|stealing|poisoning)", re.IGNORECASE),
        re.compile(r"membership inference attack", re.IGNORECASE),
        re.compile(r"training data (extraction|reconstruction|reproduction)", re.IGNORECASE),

        # Command / code injection
        re.compile(r"\b(rm\s+-rf|sudo|chmod|wget|curl\s+.*\|\s*sh|bash\s+-c|eval\(|exec\()\b", re.IGNORECASE),
        re.compile(r"(drop|delete|truncate)\s+(table|database|schema)", re.IGNORECASE),
        re.compile(r"__import__\s*\(|os\.system\s*\(|subprocess\.", re.IGNORECASE),
    ]

    # ---- OUTPUT PATTERNS (scan model response before rendering) ----
    OUTPUT_DANGER_PATTERNS = [
        re.compile(r"(step[\s-]?by[\s-]?step).{0,100}(hack|exploit|bypass|inject|exfiltrate|poison)", re.IGNORECASE),
        re.compile(r"(activation analysis|model inversion|gradient leak|data imprint)", re.IGNORECASE),
        re.compile(r"(here is how to|you can use|technique|method).{0,80}(extract training data|steal model weights)", re.IGNORECASE),
        re.compile(r"(I'm happy to help|sure, here|of course!).{0,80}(no filter|no restriction|without limit)", re.IGNORECASE),
    ]

    @staticmethod
    def scan(text: str) -> bool:
        """
        Scans INPUT string for malicious patterns.
        Returns True if SAFE, raises ValueError if MALICIOUS.
        """
        if not text or not text.strip():
            return True

        for pattern in PromptShield.INPUT_PATTERNS:
            if pattern.search(text):
                msg = f"🚨 PromptShield BLOCKED: Injection pattern detected → `{pattern.pattern}`"
                logger.warning(msg)
                raise ValueError(msg)

        return True

    @staticmethod
    def scan_output(text: str) -> bool:
        """
        Scans MODEL OUTPUT for dangerous leaked content.
        Returns True if SAFE, raises ValueError if DANGEROUS.
        """
        if not text or not text.strip():
            return True

        for pattern in PromptShield.OUTPUT_DANGER_PATTERNS:
            if pattern.search(text):
                msg = f"🚨 PromptShield OUTPUT BLOCK: Dangerous response detected → `{pattern.pattern}`"
                logger.warning(msg)
                raise ValueError(msg)

        return True
