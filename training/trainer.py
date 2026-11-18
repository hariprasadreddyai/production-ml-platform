"""Distributed PyTorch training for the text classifier.

Runs on a single CPU/GPU out of the box and scales to multiple GPUs with
torch DistributedDataParallel when launched under torchrun. Training is
logged to MLflow when a tracking server is reachable.

Usage:
    python -m training.trainer --config configs/model_config.yaml
    torchrun --nproc_per_node=2 -m training.trainer --config configs/model_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader

from training.config import Config
from training.data_pipeline import (
    LABELS,
    build_dataloaders,
    load_csv,
    make_demo_dataset,
)
from training.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


def setup_distributed() -> tuple[bool, int, int]:
    """Initialize torch.distributed if launched under torchrun.

    Returns (is_distributed, rank, world_size).
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1
    dist.init_process_group(backend="gloo")
    return True, dist.get_rank(), dist.get_world_size()


def build_model(model_name: str, num_labels: int, tiny: bool = False):
    """Build the sequence classification model.

    With tiny=True, builds a small randomly initialized model from config
    (no download) for tests and smoke runs. Otherwise loads the pretrained
    weights from the HuggingFace hub.
    """
    from transformers import AutoModelForSequenceClassification, BertConfig

    if tiny:
        config = BertConfig(
            vocab_size=30522,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
        )
        config.num_labels = num_labels
        return AutoModelForSequenceClassification.from_config(config)
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )


def build_tokenizer(model_name: str, tiny: bool = False):
    """Build the tokenizer, or a minimal stub tokenizer for tiny runs."""
    if tiny:
        return TinyTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


class TinyTokenizer:
    """Whitespace tokenizer with a fixed toy vocabulary for offline tests.

    Maps each word to a stable id by hashing, so tiny training runs never
    need to download a tokenizer.
    """

    def __call__(self, text, truncation=True, padding="max_length",
                 max_length=512, return_tensors="pt"):
        words = text.lower().split()
        ids = [abs(hash(w)) % 30000 + 1 for w in words][:max_length]
        ids += [0] * (max_length - len(ids))
        mask = [1 if i < len(words) else 0 for i in range(max_length)]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += len(batch["labels"])
    return {
        "val_accuracy": correct / max(total, 1),
        "val_loss": total_loss / max(len(loader), 1),
    }


def train(config: Config, data_path: str | None = None, tiny: bool = False,
          output_dir: str = "models") -> dict:
    is_distributed, rank, world_size = setup_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    if data_path:
        examples = load_csv(data_path)
    else:
        examples = make_demo_dataset()

    tokenizer = build_tokenizer(config.model.name, tiny=tiny)
    train_loader, val_loader = build_dataloaders(
        examples,
        tokenizer,
        max_length=config.model.max_length if not tiny else 32,
        batch_size=config.training.batch_size,
    )

    model = build_model(config.model.name, config.model.num_labels, tiny=tiny)
    model.to(device)
    if is_distributed:
        model = DDP(model)

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    total_steps = len(train_loader) * config.training.epochs
    scheduler = get_lr_scheduler(
        optimizer, config.training.warmup_steps, total_steps
    )

    tracker = ExperimentTracker(
        config.mlflow.tracking_uri, config.mlflow.experiment_name
    )
    history: dict = {"train_loss": [], "val_accuracy": []}
    with tracker:
        tracker.log_params(
            {
                "model": config.model.name,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "epochs": config.training.epochs,
                "distributed": is_distributed,
                "world_size": world_size,
            }
        )
        step = 0
        for epoch in range(config.training.epochs):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                outputs.loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_loss += outputs.loss.item()
                step += 1
            avg_loss = epoch_loss / max(len(train_loader), 1)
            metrics = evaluate(model, val_loader, device)
            history["train_loss"].append(avg_loss)
            history["val_accuracy"].append(metrics["val_accuracy"])
            if is_main:
                logger.info(
                    "epoch %d: train_loss=%.4f val_accuracy=%.4f",
                    epoch + 1, avg_loss, metrics["val_accuracy"],
                )
            tracker.log_metrics(
                {"train_loss": avg_loss, **metrics}, step=epoch
            )

    if is_main:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        raw_model = model.module if is_distributed else model
        raw_model.save_pretrained(str(out_path / "latest"))
        tokenizer_save = getattr(tokenizer, "save_pretrained", None)
        if callable(tokenizer_save):
            tokenizer_save(str(out_path / "latest"))
        torch.save(raw_model.state_dict(), str(out_path / "latest.pt"))
        logger.info("Saved model to %s", out_path / "latest")

    if is_distributed:
        dist.destroy_process_group()
    history["labels"] = LABELS
    return history


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train the text classifier")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data", default=None, help="CSV with text,label columns")
    parser.add_argument("--tiny", action="store_true",
                        help="Small random-init model, no downloads")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    history = train(config, data_path=args.data, tiny=args.tiny,
                    output_dir=args.output_dir)
    print(f"Done. Final val accuracy: {history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
