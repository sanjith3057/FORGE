import logging
import os
from src.config import ForgeConfig

try:
    from unsloth import FastLanguageModel
except ImportError:
    raise ImportError("Unsloth is required. Please install via: pip install unsloth[colab-new]")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ForgeMerge")

def merge_adapter():
    config = ForgeConfig.rtx_2050()
    adapter_path = os.path.join(config.output_dir, "final_adapter")
    
    if not os.path.exists(adapter_path):
        logger.error(f"Adapter not found at {adapter_path}. Run training first.")
        return
        
    logger.info(f"Loading base model and adapter from {adapter_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    
    # Load adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
    )
    model.load_adapter(adapter_path)
    
    logger.info("Merging adapter into base model weights...")
    
    # Unsloth supports fast saving of merged 4-bit models directly
    # We save in 4-bit config to preserve the low VRAM footprint
    output_dir = os.path.join(config.merged_dir, "final_merged_4bit")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving merged model to {output_dir}")
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_4bit_forced")
    
    logger.info("Merge complete. Model is ready for standalone inference.")

if __name__ == "__main__":
    merge_adapter()
