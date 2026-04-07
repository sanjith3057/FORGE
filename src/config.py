from dataclasses import dataclass, field

@dataclass
class ForgeConfig:
    # Model
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length: int = 1024       # ← Aggressive: 1024 not 2048 (halves memory)
    dtype: str = "float16"
    load_in_4bit: bool = True
    use_double_quant: bool = True     # ← Double quantisation for extra VRAM savings
    bnb_4bit_quant_type: str = "nf4"  # ← NF4 > FP4 for LLM weights

    # LoRA — Lean config for 4 GB VRAM
    lora_r: int = 8                   # ← r=8 not 16 (fewer trainable params)
    lora_alpha: int = 16              # ← alpha = 2*r
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"  # ← Attention only (skip MLP for VRAM)
    ])

    # Training — RTX 2050 aggressive defaults
    num_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1   # ← Batch 1 (minimum possible)
    gradient_accumulation_steps: int = 8   # ← Effective batch = 8
    warmup_steps: int = 10
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    logging_steps: int = 5
    save_steps: int = 50
    gradient_checkpointing: bool = True    # ← Trades compute for VRAM
    optim: str = "adamw_8bit"              # ← 8-bit optimizer saves ~1 GB
    fp16: bool = True
    bf16: bool = False                     # ← RTX 2050 = Ampere, but fp16 is safer

    # Paths
    train_data: str = "data/train.jsonl"
    valid_data: str = "data/valid.jsonl"
    test_data: str = "data/test.jsonl"
    output_dir: str = "outputs/adapters"
    merged_dir: str = "outputs/merged"
    log_dir: str = "logs"

    @classmethod
    def colab_t4(cls) -> "ForgeConfig":
        """Relaxed config for Colab T4 (16 GB VRAM)."""
        return cls(
            max_seq_length=2048,
            lora_r=16,
            lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
        )

    @classmethod
    def rtx_2050(cls) -> "ForgeConfig":
        """Aggressive config for RTX 2050 (4 GB VRAM). Default."""
        return cls()  # defaults are already tuned for low-VRAM
