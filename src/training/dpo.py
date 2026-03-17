"""
DPO (Direct Preference Optimization) fine-tuning for human-value alignment.
Safer alternative to PPO-based RLHF — more stable training, no reward model needed.
"""
import torch
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from peft import LoraConfig
import mlflow


def run_dpo(model, tokenizer, preference_dataset: Dataset, config: dict):
    """
    Fine-tune the VLM using DPO on a preference dataset.

    preference_dataset must have columns:
        - prompt: str
        - chosen: str   (preferred response)
        - rejected: str (dispreferred response)
    """
    cfg = config["training"]["dpo"]

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    training_args = DPOConfig(
        beta=cfg["beta"],
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=3,
        fp16=True,
        logging_steps=10,
        output_dir="checkpoints/dpo",
        report_to="mlflow",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # ref model auto-created from base when using PEFT
        args=training_args,
        train_dataset=preference_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    mlflow.set_experiment("vision-lang-dpo")
    with mlflow.start_run():
        mlflow.log_params(cfg)
        trainer.train()
        trainer.save_model("checkpoints/dpo_final")
        mlflow.log_artifacts("checkpoints/dpo_final")
