from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.platform import profile_target_column, profile_data, train_model


def _fake_register_experiment(**kwargs):
    return {
        "id": "test-experiment",
        "created_at": "2026-05-28T00:00:00+00:00",
        "dataset_path": kwargs["dataset_path"],
        "target_column": kwargs["target_column"],
        "task_type": kwargs["task_type"],
        "total_rows": kwargs["total_rows"],
        "total_columns": kwargs["total_columns"],
        "best_model_name": kwargs["best_model_name"],
        "metric_name": kwargs["metric_name"],
        "metric_value": kwargs["metric_value"],
        "model_path": "memory",
        "metadata": kwargs["metadata"],
    }


class TargetTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_dataset(self, frame: pd.DataFrame, name: str = "dataset.csv") -> Path:
        path = self.temp_path / name
        frame.to_csv(path, index=False)
        return path

    def test_profile_blocks_identifier_targets(self) -> None:
        frame = pd.DataFrame(
            {
                "project_no": [f"P-{idx:04d}" for idx in range(60)],
                "status_group": ["complete"] * 30 + ["active"] * 30,
                "budget_cost": list(range(60)),
            }
        )

        profile = profile_target_column(frame, "project_no")

        self.assertFalse(profile["is_trainable"])
        self.assertEqual(profile["quality"], "blocked")
        self.assertTrue(any("identifier" in reason for reason in profile["reasons"]))

    def test_profile_data_returns_filtered_target_candidates(self) -> None:
        frame = pd.DataFrame(
            {
                "project_no": [f"P-{idx:04d}" for idx in range(90)],
                "status_group": ["complete"] * 45 + ["cancelled"] * 30 + ["active"] * 15,
                "budget_cost": [float(idx * 10) for idx in range(90)],
            }
        )
        path = self._write_dataset(frame)

        profile = profile_data(path)

        self.assertIn("target_profiles", profile)
        self.assertIn("status_group", profile["target_candidates"])
        self.assertIn("budget_cost", profile["target_candidates"])
        self.assertNotIn("project_no", profile["target_candidates"])

    def test_classification_training_succeeds_for_grouped_target(self) -> None:
        frame = pd.DataFrame(
            {
                "status_group": ["complete"] * 45 + ["cancelled"] * 30 + ["active"] * 15,
                "budget_revenue": [1000 + idx * 3 for idx in range(90)],
                "budget_cost": [500 + idx * 2 for idx in range(90)],
                "related_contacts": ["Team A", "Team B", "Team C"] * 30,
            }
        )
        path = self._write_dataset(frame)

        with patch("app.platform.register_experiment", side_effect=_fake_register_experiment):
            result = train_model(dataset_path=path, target_column="status_group")

        self.assertEqual(result["experiment"]["task_type"], "classification")
        self.assertIn("f1_weighted", result["best_metrics"])

    def test_regression_training_succeeds_for_numeric_target(self) -> None:
        frame = pd.DataFrame(
            {
                "budget_cost": [float(250 + idx * 8) for idx in range(90)],
                "budget_revenue": [float(600 + idx * 12) for idx in range(90)],
                "scaffold_height": [float((idx % 9) + 1) for idx in range(90)],
                "related_contacts": ["Team A", "Team B", "Team C"] * 30,
            }
        )
        path = self._write_dataset(frame)

        with patch("app.platform.register_experiment", side_effect=_fake_register_experiment):
            result = train_model(dataset_path=path, target_column="budget_cost", task_type="regression")

        self.assertEqual(result["experiment"]["task_type"], "regression")
        self.assertIn("r2", result["best_metrics"])

    def test_forced_regression_rejects_non_numeric_text_target(self) -> None:
        frame = pd.DataFrame(
            {
                "status_group": ["complete"] * 45 + ["cancelled"] * 30 + ["active"] * 15,
                "budget_revenue": [1000 + idx * 3 for idx in range(90)],
                "related_contacts": ["Team A", "Team B", "Team C"] * 30,
            }
        )
        path = self._write_dataset(frame)

        with self.assertRaisesRegex(ValueError, "not trainable as regression"):
            train_model(dataset_path=path, target_column="status_group", task_type="regression")


if __name__ == "__main__":
    unittest.main()
