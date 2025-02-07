import pytest
import pandas as pd
from unittest.mock import Mock
from ragaai_catalyst import RedTeaming


class TestRedTeaming:

    @pytest.fixture
    def red_teaming(self, monkeypatch):
        """Fixture to create a RedTeaming instance with a mock API key."""
        monkeypatch.setenv("GISKARD_API_KEY", "test_api_key")
        return RedTeaming()

    def test_invalid_model_type(self, red_teaming):
        """Test that an invalid model_type raises a ValueError."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            red_teaming.run_scan(
                model=Mock(),
                model_type="invalid_type",
                name="Test Model",
                description="A test model"
            )

    def test_invalid_scan_metric(self, red_teaming):
        """Test that an invalid scan metric raises a ValueError."""
        with pytest.raises(ValueError, match="Invalid scan metrics"):
            red_teaming.run_scan(
                model=Mock(),
                model_type="text_generation",
                name="Test Model",
                description="A test model",
                only=["invalid_metric"]
            )

    def test_successful_scan(self, red_teaming, monkeypatch):
        """Test a successful scan by mocking the giskard scan method."""
        mock_report = Mock()
        mock_report.to_dataframe.return_value = pd.DataFrame({"result": ["pass"]})

        mock_giskard = Mock()
        mock_giskard.scan.return_value = mock_report
        monkeypatch.setattr("giskard.scan", mock_giskard.scan)

        df = red_teaming.run_scan(
            model=Mock(),
            model_type="text_generation",
            name="Test Model",
            description="A test model"
        )

        assert isinstance(df, pd.DataFrame)
        assert "result" in df.columns
        assert df.iloc[0]["result"] == "pass"

    def test_scan(self):
        def invoke(query: dict):
            return f"Dummy answer for '{query['query']}'"

        def model_predict(df: pd.DataFrame):
            return [invoke({"query": question}) for question in df["question"]]

        red_teaming = RedTeaming()
        scan_df = red_teaming.run_scan(
            model=model_predict,
            model_type="text_generation",
            name="demo_name",
            description="demo_description"
        )

        assert not scan_df.empty, "The scan DataFrame should not be empty."
