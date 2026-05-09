"""Unit tests for Flowise profile compaction and failure handling."""

from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from src.ai.flowise_client import (
    FLOWISE_RESPONSE_PREFIX,
    build_flowise_request_payload,
    build_default_flowise_metadata,
    build_flowise_file_preview,
    validate_flowise_profile_text,
    query_flowise_agent,
)


class TestFlowiseClient(unittest.TestCase):
    def test_build_flowise_file_preview_is_compact_and_truncates_long_text(self):
        long_text = "A" * 400
        dataframe = pd.DataFrame(
            {
                "label": ["ham", "spam"],
                "message": [long_text, "short text"],
                "category": ["alpha", "beta"],
            }
        )
        profile = {
            "rows": 2,
            "columns": 3,
            "column_names": ["label", "message", "category"],
            "data_types": {"label": "object", "message": "object", "category": "object"},
            "missing_values": {"label": 0, "message": 0, "category": 0},
            "duplicate_rows": 0,
            "numeric_columns": [],
            "categorical_columns": ["label", "category"],
            "text_columns": ["message"],
            "datetime_columns": [],
        }
        ml_recommendation = {
            "recommended_problem_type": "NLP/Text Classification",
            "problem_type_reason": "A message column and label column are present.",
            "selected_target_column": "label",
            "suggested_target_column": "label",
            "algorithms": [
                {"name": "Naive Bayes", "reason": "Good baseline for text classification."}
            ],
        }

        preview = build_flowise_file_preview(
            dataframe,
            target_column="label",
            profile=profile,
            ml_recommendation=ml_recommendation,
            file_name="spam.csv",
            max_rows=10,
        )
        preview_json = json.loads(preview)

        self.assertEqual(preview_json["original_file_name"], "spam.csv")
        self.assertEqual(preview_json["preview_rows_sent"], 2)
        self.assertFalse(preview_json["full_dataset_sent_to_flowise"])
        self.assertEqual(len(preview_json["sample_rows"]), 2)
        self.assertLessEqual(len(preview_json["sample_rows"][0]["message"]), 120)
        self.assertNotIn(long_text, preview)

    def test_build_flowise_file_preview_limits_sample_columns_for_wide_datasets(self):
        dataframe = pd.DataFrame(
            [{f"column_{index}": index for index in range(40)} for _ in range(3)]
        )
        profile = {
            "rows": 3,
            "columns": 40,
            "column_names": list(dataframe.columns),
            "data_types": {column: "int64" for column in dataframe.columns},
            "missing_values": {column: 0 for column in dataframe.columns},
            "duplicate_rows": 0,
            "numeric_columns": list(dataframe.columns),
            "categorical_columns": [],
            "text_columns": [],
            "datetime_columns": [],
        }
        preview = build_flowise_file_preview(
            dataframe,
            profile=profile,
            ml_recommendation={},
            file_name="wide.csv",
            max_rows=10,
        )
        preview_json = json.loads(preview)

        self.assertEqual(len(preview_json["sample_row_columns_included"]), 25)
        self.assertEqual(preview_json["sample_row_omitted_column_count"], 15)

    def test_build_flowise_request_payload_embeds_profile_in_question_only(self):
        profile_text = json.dumps(
            {
                "shape": {"rows": 891, "columns": 12},
                "column_names": ["Survived", "Pclass"],
                "sample_rows": [{"Survived": 0, "Pclass": 3}],
            }
        )

        payload, metadata = build_flowise_request_payload(
            "Tell me dataset shape",
            profile_text,
        )

        self.assertIn("Python-generated dataset profile:", payload["question"])
        self.assertIn("\"rows\": 891", payload["question"])
        self.assertIn("Important: Use this profile as the dataset source.", payload["question"])
        self.assertNotIn("file", payload)
        self.assertEqual(payload["overrideConfig"], {})
        self.assertTrue(metadata["profile_sent_to_flowise"])
        self.assertEqual(metadata["profile_keys_sent"], ["column_names", "sample_rows", "shape"])
        self.assertEqual(metadata["preview_rows_sent"], 1)
        self.assertFalse(metadata["full_dataset_sent_to_flowise"])

    def test_validate_flowise_profile_text_rejects_incomplete_profile(self):
        is_valid, parsed = validate_flowise_profile_text(json.dumps({"shape": {"rows": 1, "columns": 2}}))

        self.assertFalse(is_valid)
        self.assertEqual(parsed["shape"]["rows"], 1)

    @patch("src.ai.flowise_client.requests.post", side_effect=Exception("network down"))
    def test_query_flowise_agent_handles_failure_safely(self, mocked_post):
        result = query_flowise_agent(
            "Explain the dataset",
            file_summary=json.dumps(
                {
                    "shape": {"rows": 2, "columns": 2},
                    "column_names": ["a", "b"],
                    "sample_rows": [1, 2],
                }
            ),
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Flowise Agent is currently unavailable. The Python cleaning report is still available.")
        self.assertEqual(result["metadata"]["flowise_status"], "error")
        self.assertEqual(result["metadata"]["preview_rows_sent"], 2)
        self.assertIn("question", result["payload"])
        self.assertNotIn("file", result["payload"])
        mocked_post.assert_called_once()

    @patch("src.ai.flowise_client.requests.post")
    def test_query_flowise_agent_prefixes_response_with_profile_limitation(self, mocked_post):
        mocked_response = Mock()
        mocked_response.status_code = 200
        mocked_response.json.return_value = {"text": "The dataset is suitable for classification."}
        mocked_post.return_value = mocked_response

        result = query_flowise_agent(
            "Explain the dataset",
            file_summary=json.dumps(
                {
                    "shape": {"rows": 2, "columns": 2},
                    "column_names": ["a", "b"],
                    "sample_rows": [{}, {}],
                }
            ),
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["answer"].startswith(FLOWISE_RESPONSE_PREFIX))
        self.assertEqual(result["metadata"]["preview_rows_sent"], 2)
        self.assertEqual(result["metadata"]["flowise_status"], "success")
        self.assertIn("question", result["payload"])
        self.assertNotIn("file", result["payload"])

    def test_default_flowise_metadata_is_report_safe(self):
        metadata = build_default_flowise_metadata()

        self.assertFalse(metadata["flowise_called"])
        self.assertFalse(metadata["profile_sent_to_flowise"])
        self.assertEqual(metadata["profile_keys_sent"], [])
        self.assertFalse(metadata["full_dataset_sent_to_flowise"])
        self.assertEqual(metadata["preview_rows_sent"], 0)
        self.assertEqual(metadata["flowise_status"], "skipped")
        self.assertIsNone(metadata["flowise_error_message"])


if __name__ == "__main__":
    unittest.main()
