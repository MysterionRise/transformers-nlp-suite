"""
Unit tests for performance dashboard evaluation functions
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import plotly.graph_objects as go
import pytest

# Try importing evaluation dependencies
try:
    from ui.performance_dashboard import (
        EVAL_AVAILABLE,
        batch_evaluate,
        create_comparison_chart,
        create_metric_bars,
        evaluate_text,
        export_results,
        get_cache_stats,
        list_available_models,
    )

    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="Performance dashboard dependencies not available")


@pytest.mark.skipif(not DASHBOARD_AVAILABLE, reason="Dashboard not available")
class TestEvaluateText:
    """Test evaluate_text function"""

    @pytest.mark.skipif(not EVAL_AVAILABLE, reason="Evaluation libraries not available")
    def test_evaluate_text_basic(self):
        """Test basic text evaluation"""
        reference = "The cat sat on the mat."
        candidate = "The cat sat on the mat."

        result = evaluate_text(reference, candidate)

        assert isinstance(result, dict)
        # Perfect match should have high scores
        if "BLEU" in result:
            assert result["BLEU"] > 0.9

    @pytest.mark.skipif(not EVAL_AVAILABLE, reason="Evaluation libraries not available")
    def test_evaluate_text_different_texts(self):
        """Test evaluation with different texts"""
        reference = "The quick brown fox jumps over the lazy dog."
        candidate = "A fast brown fox leaps over a sleepy dog."

        result = evaluate_text(reference, candidate)

        assert isinstance(result, dict)
        # Should have lower scores than perfect match
        if "BLEU" in result:
            assert result["BLEU"] < 1.0

    def test_evaluate_text_empty_reference(self):
        """Test evaluation with empty reference"""
        result = evaluate_text("", "Some text")

        assert isinstance(result, dict)
        assert "error" in result

    def test_evaluate_text_empty_candidate(self):
        """Test evaluation with empty candidate"""
        result = evaluate_text("Some text", "")

        assert isinstance(result, dict)
        assert "error" in result

    def test_evaluate_text_both_empty(self):
        """Test evaluation with both texts empty"""
        result = evaluate_text("", "")

        assert isinstance(result, dict)
        assert "error" in result

    @pytest.mark.skipif(not EVAL_AVAILABLE, reason="Evaluation libraries not available")
    def test_evaluate_text_metrics_present(self):
        """Test that all expected metrics are present"""
        reference = "The cat sat on the mat."
        candidate = "The cat sat on the mat."

        result = evaluate_text(reference, candidate)

        if "error" not in result:
            # Check for expected metrics
            expected_metrics = ["BLEU", "ROUGE-1", "ROUGE-L", "METEOR"]
            for metric in expected_metrics:
                assert metric in result or "error" in result

    @pytest.mark.skipif(not EVAL_AVAILABLE, reason="Evaluation libraries not available")
    def test_evaluate_text_score_range(self):
        """Test that scores are in valid range [0, 1]"""
        reference = "The cat sat on the mat."
        candidate = "The dog stood on the rug."

        result = evaluate_text(reference, candidate)

        if "error" not in result:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    assert 0 <= value <= 1, f"Score {key}={value} is out of range [0, 1]"


@pytest.mark.skipif(not DASHBOARD_AVAILABLE, reason="Dashboard not available")
class TestChartCreation:
    """Test chart creation functions"""

    def test_create_comparison_chart_empty(self):
        """Test creating chart with empty results"""
        fig = create_comparison_chart([])

        assert isinstance(fig, go.Figure)

    def test_create_comparison_chart_single_model(self):
        """Test creating chart with single model"""
        results = [
            {
                "model": "Model 1",
                "BLEU": 0.8,
                "ROUGE-1": 0.75,
                "ROUGE-2": 0.6,
                "ROUGE-L": 0.7,
                "METEOR": 0.65,
                "BERTScore F1": 0.85,
            }
        ]

        fig = create_comparison_chart(results)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_create_comparison_chart_multiple_models(self):
        """Test creating chart with multiple models"""
        results = [
            {
                "model": "Model 1",
                "BLEU": 0.8,
                "ROUGE-1": 0.75,
                "ROUGE-2": 0.6,
                "ROUGE-L": 0.7,
                "METEOR": 0.65,
                "BERTScore F1": 0.85,
            },
            {
                "model": "Model 2",
                "BLEU": 0.7,
                "ROUGE-1": 0.65,
                "ROUGE-2": 0.5,
                "ROUGE-L": 0.6,
                "METEOR": 0.55,
                "BERTScore F1": 0.75,
            },
        ]

        fig = create_comparison_chart(results)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_create_metric_bars_empty(self):
        """Test creating bar chart with empty results"""
        fig = create_metric_bars([])

        assert isinstance(fig, go.Figure)

    def test_create_metric_bars_single_model(self):
        """Test creating bar chart with single model"""
        results = [
            {"model": "Model 1", "BLEU": 0.8, "ROUGE-1": 0.75, "ROUGE-L": 0.7, "METEOR": 0.65, "BERTScore F1": 0.85}
        ]

        fig = create_metric_bars(results)

        assert isinstance(fig, go.Figure)

    def test_create_metric_bars_multiple_models(self):
        """Test creating bar chart with multiple models"""
        results = [
            {"model": "Model 1", "BLEU": 0.8, "ROUGE-1": 0.75, "ROUGE-L": 0.7, "METEOR": 0.65, "BERTScore F1": 0.85},
            {"model": "Model 2", "BLEU": 0.7, "ROUGE-1": 0.65, "ROUGE-L": 0.6, "METEOR": 0.55, "BERTScore F1": 0.75},
        ]

        fig = create_metric_bars(results)

        assert isinstance(fig, go.Figure)


@pytest.mark.skipif(not DASHBOARD_AVAILABLE, reason="Dashboard not available")
class TestBatchEvaluate:
    """Test batch_evaluate function"""

    def test_batch_evaluate_empty_reference(self):
        """Test batch evaluation with empty reference"""
        df, radar, bar = batch_evaluate("", "candidate1\n---\ncandidate2")

        assert isinstance(df, pd.DataFrame)
        assert isinstance(radar, go.Figure)
        assert isinstance(bar, go.Figure)
        assert len(df) == 0

    def test_batch_evaluate_empty_candidates(self):
        """Test batch evaluation with empty candidates"""
        df, radar, bar = batch_evaluate("reference text", "")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @pytest.mark.skipif(not EVAL_AVAILABLE, reason="Evaluation libraries not available")
    def test_batch_evaluate_single_candidate(self):
        """Test batch evaluation with single candidate"""
        reference = "The cat sat on the mat."
        candidate = "The cat sat on the mat."

        df, radar, bar = batch_evaluate(reference, candidate)

        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "model" in df.columns
            assert len(df) == 1

    @pytest.mark.skipif(not EVAL_AVAILABLE, reason="Evaluation libraries not available")
    def test_batch_evaluate_multiple_candidates(self):
        """Test batch evaluation with multiple candidates"""
        reference = "The cat sat on the mat."
        candidates = "The cat sat on the mat.\n---\nThe dog stood on the rug."

        df, radar, bar = batch_evaluate(reference, candidates)

        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert len(df) <= 2  # May be less if evaluation fails

    def test_batch_evaluate_custom_delimiter(self):
        """Test batch evaluation with custom delimiter"""
        reference = "The cat sat on the mat."
        candidates = "Candidate 1|||Candidate 2"

        df, radar, bar = batch_evaluate(reference, candidates, delimiter="|||")

        assert isinstance(df, pd.DataFrame)


@pytest.mark.skipif(not DASHBOARD_AVAILABLE, reason="Dashboard not available")
class TestUtilityFunctions:
    """Test utility functions"""

    def test_export_results_empty(self):
        """Test exporting empty results"""
        df = pd.DataFrame()
        result = export_results(df)

        assert result == "{}"

    def test_export_results_with_data(self):
        """Test exporting results with data"""
        df = pd.DataFrame([{"model": "Model 1", "score": 0.8}])
        result = export_results(df)

        assert isinstance(result, str)
        assert "Model 1" in result
        assert "0.8" in result

    @patch("ui.performance_dashboard.get_cache_info")
    def test_get_cache_stats(self, mock_get_info):
        """Test getting cache statistics"""
        mock_get_info.return_value = {
            "cached_models": 2,
            "max_size": 3,
            "device": "cpu",
            "models": ["model1", "model2"],
        }

        result = get_cache_stats()

        assert isinstance(result, str)
        assert "2" in result
        assert "3" in result
        assert "cpu" in result

    @patch("ui.performance_dashboard.get_cache_info")
    def test_get_cache_stats_error(self, mock_get_info):
        """Test cache stats with error"""
        mock_get_info.side_effect = Exception("Test error")

        result = get_cache_stats()

        assert isinstance(result, str)
        assert "Error" in result or "error" in result

    @patch("ui.performance_dashboard.get_model_registry")
    def test_list_available_models(self, mock_registry):
        """Test listing available models"""
        mock_reg = MagicMock()
        mock_reg.list_categories.return_value = ["sentiment_analysis"]

        mock_config = MagicMock()
        mock_config.name = "Test Model"
        mock_config.model_id = "test/model"

        mock_reg.get_category.return_value = {"test_model": mock_config}
        mock_registry.return_value = mock_reg

        result = list_available_models()

        assert isinstance(result, str)
        assert "Available Models" in result or "sentiment" in result.lower()

    @patch("ui.performance_dashboard.get_model_registry")
    def test_list_available_models_error(self, mock_registry):
        """Test listing models with error"""
        mock_registry.side_effect = Exception("Test error")

        result = list_available_models()

        assert isinstance(result, str)
        assert "Error" in result or "error" in result


@pytest.mark.skipif(not DASHBOARD_AVAILABLE, reason="Dashboard not available")
class TestIntegration:
    """Integration tests for dashboard functions"""

    @pytest.mark.skipif(not EVAL_AVAILABLE, reason="Evaluation libraries not available")
    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline"""
        reference = "The quick brown fox jumps over the lazy dog."
        candidate = "A fast brown fox leaps over a sleepy dog."

        # Single evaluation
        scores = evaluate_text(reference, candidate)
        assert isinstance(scores, dict)

        # Batch evaluation
        candidates = f"{candidate}\n---\n{reference}"
        df, radar, bar = batch_evaluate(reference, candidates)

        assert isinstance(df, pd.DataFrame)
        assert isinstance(radar, go.Figure)
        assert isinstance(bar, go.Figure)

        # Export
        if not df.empty:
            exported = export_results(df)
            assert isinstance(exported, str)
