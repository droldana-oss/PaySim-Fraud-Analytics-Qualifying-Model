import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling_utils import (  # noqa: E402
    MODEL_FEATURES,
    build_random_forest_pipeline,
    engineer_features,
    load_transactions,
    temporal_split,
    training_sample,
    xy,
)


class ModelingUtilsTests(unittest.TestCase):
    def make_raw(self):
        return pd.DataFrame({
            "step": [1, 2, 324, 325, 379, 400],
            "type": ["TRANSFER", "PAYMENT", "CASH_OUT", "PAYMENT", "TRANSFER", "CASH_IN"],
            "amount": [100.0, 5.0, 200.0, 10.0, 300.0, 8.0],
            "oldbalanceOrg": [100.0, 50.0, 200.0, 50.0, 300.0, 40.0],
            "newbalanceOrig": [0.0, 45.0, 0.0, 40.0, 0.0, 32.0],
            "oldbalanceDest": [0.0] * 6,
            "newbalanceDest": [100.0, 5.0, 200.0, 10.0, 300.0, 8.0],
            "isFraud": [1, 0, 1, 0, 1, 0],
            "isFlaggedFraud": [0, 0, 0, 0, 0, 0],
        })

    def test_features_exclude_target_alert_time_and_identifiers(self):
        data = engineer_features(self.make_raw())
        self.assertEqual(data.loc[0, "is_origin_balance_zero"], 1)
        for forbidden in ["isFraud", "isFlaggedFraud", "step", "nameOrig", "nameDest"]:
            self.assertNotIn(forbidden, MODEL_FEATURES)

    def test_temporal_split_and_training_sample_preserve_fraud(self):
        train, validation, test = temporal_split(engineer_features(self.make_raw()))
        self.assertEqual(train["step"].max(), 2)
        self.assertEqual(validation["step"].min(), 324)
        self.assertEqual(test["step"].min(), 379)
        sampled = training_sample(train, ratio=1)
        self.assertEqual(int(sampled["isFraud"].sum()), int(train["isFraud"].sum()))
        self.assertEqual(len(sampled), 2)

    def test_pipeline_accepts_new_transaction_type(self):
        data = engineer_features(self.make_raw())
        train, validation, _ = temporal_split(data)
        model = build_random_forest_pipeline()
        X_train, y_train = xy(train)
        model.fit(X_train, y_train)
        future = validation.copy()
        future["type"] = "TIPO_NUEVO"
        probability = model.predict_proba(xy(future)[0])[:, 1]
        self.assertEqual(len(probability), len(future))

    def test_missing_dataset_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            load_transactions(Path("/tmp/paysim-file-that-does-not-exist.csv"))


if __name__ == "__main__":
    unittest.main()
