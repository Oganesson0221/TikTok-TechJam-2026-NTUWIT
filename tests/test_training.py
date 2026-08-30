from pathlib import Path

import pandas as pd

from research_rec.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from research_rec.training import train_experiment
from research_rec.predict import export_predictions


def test_training_saves_validation_best_checkpoint(tmp_path: Path):
    rows = []
    for user in range(6):
        for item in range(6):
            rows.append({"user_id": user, "video_id": item, "is_click": int(item == user)})
    frame = pd.DataFrame(rows)
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    frame.to_csv(train_path, index=False)
    frame.to_csv(validation_path, index=False)
    config = ExperimentConfig(
        data=DataConfig(
            train_csv=str(train_path),
            validation_csv=str(validation_path),
            categorical_features=["user_id", "video_id"],
        ),
        model=ModelConfig(name="mf", embedding_dim=4),
        training=TrainingConfig(
            epochs=2,
            batch_size=8,
            patience=2,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_name="smoke",
            device="cpu",
        ),
    )
    result = train_experiment(config)
    assert result["epochs_completed"] == 2
    assert Path(result["checkpoint"]).is_file()
    assert (tmp_path / "checkpoints" / "smoke" / "summary.json").is_file()
    assert set(result["best_metrics"]) >= {"ndcg@10", "recall@50", "loss"}

    export_path = tmp_path / "submission.csv"
    exported = export_predictions(result["checkpoint"], validation_path, export_path, top_k=3, device_name="cpu")
    prediction_frame = pd.read_csv(export_path)
    assert exported["label_columns_read"] == []
    assert list(prediction_frame.columns) == ["user_id", "video_id", "score", "rank"]
    assert prediction_frame.groupby("user_id").size().max() == 3


def test_multitask_training_uses_auxiliary_labels(tmp_path: Path):
    rows = []
    for user in range(4):
        for item in range(5):
            clicked = int(item == user)
            rows.append(
                {
                    "user_id": user,
                    "video_id": item,
                    "is_click": clicked,
                    "long_view": clicked,
                    "is_like": int(clicked and user % 2 == 0),
                }
            )
    frame = pd.DataFrame(rows)
    train_path = tmp_path / "multi_train.csv"
    validation_path = tmp_path / "multi_validation.csv"
    frame.to_csv(train_path, index=False)
    frame.to_csv(validation_path, index=False)
    config = ExperimentConfig(
        data=DataConfig(
            train_csv=str(train_path),
            validation_csv=str(validation_path),
            categorical_features=["user_id", "video_id"],
            auxiliary_label_columns=["long_view", "is_like"],
        ),
        model=ModelConfig(name="multitask_dcn", embedding_dim=4, hidden_dims=[8, 4]),
        training=TrainingConfig(
            epochs=1,
            batch_size=8,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_name="multitask",
            device="cpu",
        ),
    )
    result = train_experiment(config)
    assert result["model"] == "multitask_dcn"
    assert Path(result["checkpoint"]).is_file()
