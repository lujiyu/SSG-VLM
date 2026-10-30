import os
import torch
from datasets import load_dataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from trl import SFTTrainer, SFTConfig


# ============================================================
# 1. Path
# ============================================================

MODEL_PATH = "./models/Qwen3.5-4B"

DATA_PATH = "./data/sample_50000_ratio/train_5.0.jsonl"

OUTPUT_DIR = "./outputs/qwen3.5-4b-safety-lora-unsloth_ration0918c75" 



# ============================================================
# 2. Dataset
# ============================================================

TRAIN_DATA_PATH = "./data/sample_50000_ratio/train_5.0.jsonl"
VALID_DATA_PATH = "./data/sample_50000_ratio/valid_5.0.jsonl"

train_dataset = load_dataset(
    "json",
    data_files=TRAIN_DATA_PATH,
    split="train",
)

valid_dataset = load_dataset(
    "json",
    data_files=VALID_DATA_PATH,
    split="train",
)

print("=" * 60)
print("Train Dataset:")
print(train_dataset)

print("\nValid Dataset:")
print(valid_dataset)

print("\nFirst train sample:")
print(train_dataset[0])

print("\nFirst valid sample:")
print(valid_dataset[0])

# ============================================================
# 3. Load Qwen3.5-4B Vision
# ============================================================

model, processor = FastVisionModel.from_pretrained(

    model_name=MODEL_PATH,

    max_seq_length=4096, 

    # --------------------------------------------------------
    # Qwen3.5 + Unsloth
    # --------------------------------------------------------

    load_in_4bit=True,  # False True

    load_in_16bit=False,  # True False

    full_finetuning=False,
)


# ============================================================
# 4. LoRA
# ============================================================

model = FastVisionModel.get_peft_model(

    model,

    # --------------------------------------------------------
    # Vision
    # --------------------------------------------------------

    finetune_vision_layers=True,  # True False

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    finetune_language_layers=True,

    # --------------------------------------------------------
    # Attention
    # --------------------------------------------------------

    finetune_attention_modules=True,

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    finetune_mlp_modules=True,

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    r=16,

    lora_alpha=16,

    lora_dropout=0,

    bias="none",

    target_modules="all-linear",

    random_state=3407,

    use_rslora=False,

    loftq_config=None,

    #modules_to_save=["lm_head", "embed_tokens",],
)


# ============================================================
# 5. Training configuration
# ============================================================

training_args = SFTConfig(

    output_dir=OUTPUT_DIR,

    # Training
    num_train_epochs=5,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    warmup_ratio=0.05,

    # Logging
    logging_steps=10,

    # Evaluation
    eval_strategy="epoch",

    # Saving
    save_strategy="epoch",

    save_total_limit=5,

    # Precision
    bf16=True,

    # Memory
    gradient_checkpointing="unsloth",

    # Optimizer
    optim="adamw_8bit",

    # Sequence length
    max_length=4096,

    # Logging
    report_to="none",

    ddp_find_unused_parameters=False,

)


# ============================================================
# 6. Vision Data Collator
# ============================================================

data_collator = UnslothVisionDataCollator(

    model,

    processor,

    resize="max",

)


# ============================================================
# 7. Trainer
# ============================================================

trainer = SFTTrainer(

    model=model,

    train_dataset=train_dataset,

    eval_dataset=valid_dataset,

    processing_class=processor,

    data_collator=data_collator,

    args=training_args,

)


# ============================================================
# 8. Print trainable parameters
# ============================================================

print()
print("=" * 60)
print("Trainable parameters")
print("=" * 60)

model.print_trainable_parameters()

# ============================================================
# 9. Train
# ============================================================

print()
print("=" * 60)
print("Start training...")
print("=" * 60)

trainer.train()


# ============================================================
# 10. Final Evaluation
# ============================================================

print()
print("=" * 60)
print("Final validation...")
print("=" * 60)

eval_results = trainer.evaluate()

print("Validation results:")
print(eval_results)


# ============================================================
# 11. Save LoRA
# ============================================================

print()
print("=" * 60)
print("Saving LoRA...")
print("=" * 60)

model.save_pretrained(OUTPUT_DIR)

processor.save_pretrained(OUTPUT_DIR)


# ============================================================
# 12. Finished
# ============================================================

print()
print("=" * 60)
print("Training finished!")
print("=" * 60)

print("LoRA saved to:")

print(OUTPUT_DIR)
