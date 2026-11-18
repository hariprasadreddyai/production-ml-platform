"""Tests for training: config loading, data pipeline, tiny training run."""

import pandas as pd

from training.config import Config
from training.data_pipeline import (
    TextDataset,
    add_text_features,
    build_dataloaders,
    load_csv,
    make_demo_dataset,
)
from training.trainer import TinyTokenizer, train


def test_config_loads_from_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n"
        "  name: distilbert-base-uncased\n"
        "  num_labels: 3\n"
        "  max_length: 128\n"
        "training:\n"
        "  batch_size: 8\n"
        "  learning_rate: 0.00002\n"
        "  epochs: 1\n"
        "  warmup_steps: 10\n"
        "  weight_decay: 0.01\n"
        "  distributed: false\n"
        "mlflow:\n"
        "  tracking_uri: http://localhost:5000\n"
        "  experiment_name: test-run\n"
    )
    config = Config.from_yaml(path)
    assert config.model.name == "distilbert-base-uncased"
    assert config.training.batch_size == 8
    assert config.training.epochs == 1
    assert config.mlflow.experiment_name == "test-run"


def test_config_defaults():
    config = Config()
    assert config.model.num_labels == 3
    assert config.training.epochs == 3


def test_demo_dataset_has_three_classes():
    examples = make_demo_dataset(n_per_class=5)
    assert len(examples) == 15
    assert {e.label for e in examples} == {"positive", "negative", "neutral"}
    assert all(e.label_id in (0, 1, 2) for e in examples)


def test_load_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("text,label\nI love it,positive\nThis is awful,negative\n")
    examples = load_csv(path)
    assert len(examples) == 2
    assert examples[0].label == "positive"


def test_text_features():
    frame = pd.DataFrame({"text": ["Great product!", "Is it good?"]})
    out = add_text_features(frame)
    assert out.loc[0, "exclamation_count"] == 1
    assert out.loc[1, "question_count"] == 1
    assert out.loc[0, "word_count"] == 2


def test_tiny_tokenizer_output_shape():
    tokenizer = TinyTokenizer()
    encoded = tokenizer("hello world", max_length=16)
    assert encoded["input_ids"].shape == (1, 16)
    assert encoded["attention_mask"].shape == (1, 16)
    assert int(encoded["attention_mask"][0, 0]) == 1
    assert int(encoded["attention_mask"][0, 15]) == 0


def test_dataset_item_shapes():
    examples = make_demo_dataset(n_per_class=4)
    dataset = TextDataset(examples, TinyTokenizer(), max_length=16)
    item = dataset[0]
    assert item["input_ids"].shape == (16,)
    assert item["labels"].item() in (0, 1, 2)


def test_build_dataloaders_split():
    examples = make_demo_dataset(n_per_class=10)
    train_loader, val_loader = build_dataloaders(
        examples, TinyTokenizer(), max_length=16, batch_size=4
    )
    assert len(train_loader.dataset) == 24
    assert len(val_loader.dataset) == 6


def test_tiny_training_run(tmp_path):
    config = Config()
    config.training.epochs = 1
    config.training.batch_size = 8
    config.training.warmup_steps = 5
    history = train(config, tiny=True, output_dir=str(tmp_path / "models"))
    assert len(history["train_loss"]) == 1
    assert history["train_loss"][0] > 0
    assert 0.0 <= history["val_accuracy"][0] <= 1.0
    assert (tmp_path / "models" / "latest.pt").exists()
