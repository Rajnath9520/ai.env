import pytest

from app.ml.evaluation.statistics_analysis import StatisticalAnalyzer


class TestDescribe:
    def test_empty_list(self):
        stats = StatisticalAnalyzer().describe([])
        assert stats.n == 0
        assert stats.mean is None
        assert stats.std is None
        assert stats.ci_low is None

    def test_single_value_has_mean_but_no_std_or_ci(self):
        stats = StatisticalAnalyzer().describe([0.5])
        assert stats.n == 1
        assert stats.mean == pytest.approx(0.5)
        assert stats.std is None
        assert stats.ci_low is None

    def test_mean_and_std_match_known_values(self):
        stats = StatisticalAnalyzer().describe([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.mean == pytest.approx(3.0)
        assert stats.std == pytest.approx(1.5811, abs=1e-3)

    def test_confidence_interval_brackets_the_mean(self):
        stats = StatisticalAnalyzer().describe([0.2, 0.3, 0.25, 0.28, 0.31])
        assert stats.ci_low < stats.mean < stats.ci_high

    def test_wider_confidence_level_gives_wider_interval(self):
        values = [0.1, 0.5, 0.3, 0.7, 0.2]
        narrow = StatisticalAnalyzer(confidence_level=0.80).describe(values)
        wide = StatisticalAnalyzer(confidence_level=0.99).describe(values)
        assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)

    def test_more_samples_produces_a_valid_interval(self):
        small = StatisticalAnalyzer().describe([0.4, 0.6])
        large = StatisticalAnalyzer().describe([0.4, 0.6, 0.5, 0.5, 0.45, 0.55, 0.5, 0.5])
        assert small.ci_high - small.ci_low > 0
        assert large.ci_high - large.ci_low > 0


class TestCompareInsufficientData:
    def test_refuses_test_below_minimum_n(self):
        analyzer = StatisticalAnalyzer(min_n_for_test=3)
        result = analyzer.compare("A", [0.1, 0.2], "B", [0.5, 0.6])
        assert result.test_performed is False
        assert result.p_value is None
        assert result.significant is None
        assert "Insufficient samples" in result.note

    def test_still_reports_descriptive_stats_when_test_skipped(self):
        analyzer = StatisticalAnalyzer(min_n_for_test=3)
        result = analyzer.compare("A", [0.1, 0.2], "B", [0.5, 0.6])
        assert result.stats_a.mean == pytest.approx(0.15)
        assert result.stats_b.mean == pytest.approx(0.55)

    def test_exactly_at_minimum_n_runs_the_test(self):
        analyzer = StatisticalAnalyzer(min_n_for_test=3)
        result = analyzer.compare("A", [0.1, 0.15, 0.2], "B", [0.5, 0.55, 0.6])
        assert result.test_performed is True
        assert result.test_name is not None


class TestCompareWithSufficientData:
    def test_clearly_different_groups_are_significant(self):
        analyzer = StatisticalAnalyzer(min_n_for_test=3)
        group_a = [0.1, 0.12, 0.09, 0.11, 0.10]
        group_b = [0.8, 0.82, 0.79, 0.81, 0.80]
        result = analyzer.compare("low", group_a, "high", group_b)
        assert result.test_performed is True
        assert result.test_name == "Welch's t-test (unequal variance)"
        assert result.p_value < 0.01
        assert result.significant is True

    def test_identical_distributions_are_not_significant(self):
        analyzer = StatisticalAnalyzer(min_n_for_test=3)
        group_a = [0.5, 0.52, 0.48, 0.51, 0.49]
        group_b = [0.5, 0.51, 0.49, 0.50, 0.505]
        result = analyzer.compare("a", group_a, "b", group_b)
        assert result.test_performed is True
        assert result.significant is False

    def test_never_claims_significance_without_running_test(self):
        analyzer = StatisticalAnalyzer(min_n_for_test=3)
        insufficient = analyzer.compare("a", [0.1], "b", [0.9])
        assert insufficient.test_performed is False
        assert insufficient.significant is None

    def test_to_dict_is_json_serializable(self):
        import json

        analyzer = StatisticalAnalyzer(min_n_for_test=3)
        result = analyzer.compare("a", [0.1, 0.2, 0.3], "b", [0.4, 0.5, 0.6])
        json.dumps(result.to_dict())  # should not raise
