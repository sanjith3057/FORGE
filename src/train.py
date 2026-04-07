import os
import torch
import logging
from src.config import ForgeConfig
from src.data_utils import load_jsonl, validate_dataset, format_chatml
from datasets import Dataset

# For low VRAM, Unsloth is critical. If Unsloth fails to import, we fallback or exit.
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
except ImportError:
    raise ImportError("Unsloth is required. Please install via: pip install unsloth[colab-new]")

from trl import SFTTrainer
from transformers import TrainingArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ForgeTrain")

def prepare_dataset(file_path: str):
    raw_data = load_jsonl(file_path)
    if not validate_dataset(raw_data):
        raise ValueError(f"Invalid dataset: {file_path}")
    
    # Format to text strings for standard SFTTrainer
    formatted_data = [{"text": format_chatml(example)} for example in raw_data]
    return Dataset.from_list(formatted_data)

def main():
    logger.info("Initializing Forge Training Pipeline...")
    config = ForgeConfig.rtx_2050()  # Defaulting to the aggressive low-VRAM config
    
    # 1. Load Model via Unsloth
    logger.info(f"Loading Base Model: {config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None, # auto
        load_in_4bit=config.load_in_4bit
    )
    
    # 2. Attach PEFT (LoRA Adapters)
    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=0, # optimized to 0 for Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth" # heavily optimized for low VRAM
    )
    
    # 3. Datasets
    logger.info("Preparing Datasets...")
    train_dataset = prepare_dataset(config.train_data)
    
    # 4. Training Arguments
    args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config.logging_steps,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=3407,
        output_dir=config.output_dir,
        save_steps=config.save_steps,
        report_to="none" # Disable wandb for local run simplicity
    )
    
    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=2,
        packing=False, # Packing can increase peak VRAM
        args=args,
    )
    
    # 6. Train
    logger.info("Starting Training...")
    trainer_stats = trainer.train()
    
    # 7. Save Adapter
    final_output_dir = os.path.join(config.output_dir, "final_adapter")
    logger.info(f"Saving PEFT adapter to {final_output_dir}")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

if __name__ == "__main__":
    main()
