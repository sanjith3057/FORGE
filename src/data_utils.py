import json
import logging
from typing import List, Dict

logger = logging.getLogger("ForgeDataUtils")

def load_jsonl(path: str) -> List[Dict]:
    """Loads a JSONL file and returns a list of dictionaries."""
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return []

def format_chatml(example: Dict) -> str:
    """Converts a dictionary of messages into a single ChatML formatted string."""
    formatted_str = ""
    for msg in example.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        formatted_str += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return formatted_str

def validate_dataset(dataset: List[Dict]) -> bool:
    """Ensures each example in the dataset conforms to the expected ChatML list of messages."""
    for i, example in enumerate(dataset):
        if "messages" not in example:
            logger.error(f"Example {i} is missing 'messages' key.")
            return False
        
        messages = example["messages"]
        if len(messages) < 3:
            logger.error(f"Example {i} does not have system, user, and assistant roles.")
            return False
            
    logger.info(f"Dataset validated successfully. Total examples: {len(dataset)}")
    return True
