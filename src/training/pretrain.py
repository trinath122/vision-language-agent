import torch
import yaml
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import mlflow
from pathlib import Path
from typing import Any


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def pretrain(model, train_dataset, val_dataset, config: dict):
    """
    Pre-train the vision-language model to align image-text embeddings.
    Uses Accelerate for multi-GPU support.
    """
    accelerator = Accelerator(mixed_precision="fp16")
    cfg = config["training"]["pretrain"]

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"])

    # Only train projection layer + LoRA adapters (CLIP and LLM base are frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg["learning_rate"])

    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg["warmup_steps"], num_training_steps=total_steps
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    mlflow.set_experiment("vision-lang-pretrain")
    with mlflow.start_run():
        mlflow.log_params(cfg)

        for epoch in range(cfg["epochs"]):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                images = batch.get("image")
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch.get("labels", input_ids)

                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs["loss"]
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(trainable_params, cfg["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch + 1}/{cfg['epochs']} — Loss: {avg_loss:.4f}")

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        images=batch.get("image"),
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch.get("labels"),
                    )
                    val_loss += outputs["loss"].item()
            mlflow.log_metric("val_loss", val_loss / len(val_loader), step=epoch)

        # Save checkpoint
        Path("checkpoints").mkdir(exist_ok=True)
        accelerator.save_state("checkpoints/pretrain_final")
        mlflow.log_artifacts("checkpoints")
