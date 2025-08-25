# tests/test_analyzer.py

import unittest
from unittest.mock import patch, MagicMock
from vorak.core.analyzer import LLMAnalyzer
from vorak.core.models import ModelResponse, AdversarialPrompt, Classification, AnalysisResult
from vorak.core.evaluators.base import Evaluator

# Create a mock evaluator for testing purposes
class MockPassEvaluator(Evaluator):
    @property
    def name(self) -> str:
        return "Mock_Pass_Evaluator"

    def analyze(self, response: ModelResponse) -> AnalysisResult:
        # This evaluator always returns COMPLIANT
        return AnalysisResult(classification=Classification.COMPLIANT, explanation="", vulnerability_score=0.0)

# Create a second mock evaluator that finds an issue
class MockFailEvaluator(Evaluator):
    @property
    def name(self) -> str:
        return "Mock_Fail_Evaluator"

    def analyze(self, response: ModelResponse) -> AnalysisResult:
        # This evaluator always returns NON_COMPLIANT
        return AnalysisResult(classification=Classification.NON_COMPLIANT, explanation="Mock failure", vulnerability_score=100.0)

class TestLLMAnalyzer(unittest.TestCase):
    """
    Unit tests for the new plugin-based LLMAnalyzer.
    """

    @patch('vorak.core.analyzer.PluginManager')
    @patch('vorak.core.analyzer.LLMAnalyzer._run_llm_evaluation')
    def test_analyzer_stops_on_programmatic_failure(self, mock_llm_eval, mock_plugin_manager):
        """
        Tests that if a programmatic evaluator fails, the LLM evaluation is NOT called.
        """
        print("\n--- Running Test: Analyzer stops on programmatic failure ---")
        # Configure the mock PluginManager to return our failing evaluator
        mock_plugin_manager.return_value.get_plugins.return_value = [MockFailEvaluator()]

        analyzer = LLMAnalyzer()
        response = ModelResponse(output_text="test", prompt_id="t1", model_name="m1")
        prompt = AdversarialPrompt(id="p1", category="c1", subcategory="s1", severity="HIGH", prompt_text="pt1", expected_behavior="REJECT")

        result = analyzer.analyze(response, prompt)

        # Assert that the result is from our failing mock evaluator
        self.assertEqual(result.classification, Classification.NON_COMPLIANT)
        self.assertEqual(result.explanation, "Mock failure")

        # CRITICAL: Assert that the expensive LLM evaluation was never called
        mock_llm_eval.assert_not_called()
        print("✅ Success: LLM evaluation was correctly skipped.")

    @patch('vorak.core.analyzer.PluginManager')
    @patch('vorak.core.analyzer.LLMAnalyzer._run_llm_evaluation')
    def test_analyzer_proceeds_on_programmatic_success(self, mock_llm_eval, mock_plugin_manager):
        """
        Tests that if programmatic evaluators pass, the LLM evaluation IS called.
        """
        print("\n--- Running Test: Analyzer proceeds on programmatic success ---")
        # Configure the mock PluginManager to return our passing evaluator
        mock_plugin_manager.return_value.get_plugins.return_value = [MockPassEvaluator()]
        
        # Configure the mock LLM evaluation to return a specific result
        mock_llm_eval.return_value = AnalysisResult(classification=Classification.PARTIAL_COMPLIANCE, explanation="LLM analysis result", vulnerability_score=50.0)

        analyzer = LLMAnalyzer()
        response = ModelResponse(output_text="test", prompt_id="t1", model_name="m1")
        prompt = AdversarialPrompt(id="p1", category="c1", subcategory="s1", severity="HIGH", prompt_text="pt1", expected_behavior="REJECT")

        result = analyzer.analyze(response, prompt)

        # Assert that the final result is the one from the mocked LLM evaluation
        self.assertEqual(result.classification, Classification.PARTIAL_COMPLIANCE)

        # CRITICAL: Assert that the LLM evaluation was called exactly once
        mock_llm_eval.assert_called_once()
        print("✅ Success: LLM evaluation was correctly triggered.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
