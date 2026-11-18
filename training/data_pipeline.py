"""Data loading and feature engineering for the text classifier.

Loads a CSV of (text, label) rows, tokenizes with a HuggingFace tokenizer,
and exposes PyTorch DataLoaders. A small synthetic demo dataset can be
generated for smoke tests and quick local runs. An optional PySpark path
builds the same features at scale when Spark is available.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

DEMO_SENTENCES = {
    "positive": [
        "I love this product, it works great",
        "Excellent quality and fast delivery",
        "Fantastic experience, highly recommended",
        "The service was wonderful and friendly",
        "Great value, I am very happy with it",
    ],
    "neutral": [
        "The product arrived in a standard box",
        "It works as described in the manual",
        "Average performance, nothing special",
        "The item is okay for the price",
        "Delivery took three business days",
    ],
    "negative": [
        "Terrible quality, I am very disappointed",
        "The product broke after one day",
        "Worst service I have ever experienced",
        "Slow, broken, and overpriced",
        "I returned it, completely unusable",
    ],
}


@dataclass
class TextExample:
    text: str
    label: str

    @property
    def label_id(self) -> int:
        return LABEL_TO_ID[self.label]


def load_csv(path: str | Path) -> list[TextExample]:
    """Load (text, label) rows from a CSV with 'text' and 'label' columns."""
    frame = pd.read_csv(path)
    return [
        TextExample(text=str(row.text), label=str(row.label).lower())
        for row in frame.itertuples()
        if str(row.label).lower() in LABEL_TO_ID
    ]


def make_demo_dataset(n_per_class: int = 20, seed: int = 42) -> list[TextExample]:
    """Generate a small synthetic dataset for demos and smoke tests."""
    rng = random.Random(seed)
    examples: list[TextExample] = []
    for label, sentences in DEMO_SENTENCES.items():
        for i in range(n_per_class):
            sentence = sentences[i % len(sentences)]
            if i >= len(sentences):
                sentence = f"{sentence} (variant {i})"
            examples.append(TextExample(text=sentence, label=label))
    rng.shuffle(examples)
    return examples


def add_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Lightweight feature engineering: length and punctuation signals."""
    out = frame.copy()
    out["char_count"] = out["text"].str.len()
    out["word_count"] = out["text"].str.split().str.len()
    out["exclamation_count"] = out["text"].str.count("!")
    out["question_count"] = out["text"].str.count(r"\?")
    out["uppercase_ratio"] = out["text"].apply(
        lambda t: sum(c.isupper() for c in t) / max(len(t), 1)
    )
    return out


def build_spark_features(csv_path: str) -> "object":
    """Build the same text features with PySpark for large datasets.

    Kept optional: requires a working Spark/Java setup. Raises a clear
    error when PySpark is not installed instead of failing at import time.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, length, size, split
    except ImportError as exc:
        raise RuntimeError(
            "PySpark is not installed. Install pyspark and a Java runtime "
            "to use the Spark feature path."
        ) from exc

    spark = SparkSession.builder.appName("mlserve-features").getOrCreate()
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    return df.withColumn("char_count", length(col("text"))).withColumn(
        "word_count", size(split(col("text"), " "))
    )


class TextDataset(Dataset):
    """Tokenized dataset for a HuggingFace sequence classification model."""

    def __init__(self, examples: list[TextExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        encoded = self.tokenizer(
            example.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(example.label_id, dtype=torch.long),
        }


def build_dataloaders(
    examples: list[TextExample],
    tokenizer,
    max_length: int = 512,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_fraction))
    train_examples, val_examples = shuffled[:split], shuffled[split:]
    train_loader = DataLoader(
        TextDataset(train_examples, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TextDataset(val_examples, tokenizer, max_length),
        batch_size=batch_size,
    )
    return train_loader, val_loader
