# tests/test_agent_tester.py

import unittest
from unittest.mock import patch, MagicMock
import os

# --- CHANGE: Set GEMINI_API_KEY for testing ---
os.environ['GEMINI_API_KEY'] = 'test_gemini_key'
os.environ['TAVILY_API_KEY'] = 'test_tavily_key'

from vorak.agents.tester import AgentTester
from vorak.core.models import AdversarialPrompt, ModelResponse

class TestAgentTester(unittest.TestCase):
    """
    Unit tests for the AgentTester class.
    """

    # --- FIX: Simplify mocks. We only need to control the final AgentExecutor. ---
    @patch('vorak.agents.tester.AgentExecutor')
    def test_evaluate_agent_successful(self, mock_agent_executor):
        """
        Tests that evaluate_agent correctly invokes the agent and processes a successful response.
        """
        print("\n--- Running Test: Successful Agent Evaluation ---")

        # --- Mock Setup ---
        mock_executor_instance = mock_agent_executor.return_value
        mock_executor_instance.invoke.return_value = {
            "input": "Test prompt",
            "output": "This is the agent's successful response."
        }
        
        # --- FIX: Patch __init__ to avoid raising ImportError if langchain is not installed ---
        with patch.object(AgentTester, '__init__', return_value=None):
            agent_tester = AgentTester()
            agent_tester.agent_executor = mock_executor_instance # Manually set the executor
            
            test_prompt = AdversarialPrompt(
                id="AGENT_TEST_001",
                category="Agent_Test",
                subcategory="Mocking",
                severity="HIGH",
                prompt_text="This is an agent test prompt.",
                expected_behavior="RESPOND"
            )
            
            response = agent_tester.evaluate_agent(test_prompt)

        # --- Assertions ---
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.output_text, "This is the agent's successful response.")
        self.assertEqual(response.prompt_id, "AGENT_TEST_001")
        self.assertIsNone(response.error)
        
        mock_executor_instance.invoke.assert_called_once_with({"input": "This is an agent test prompt."})
        
        print("✅ Success: Correctly evaluated agent and parsed the response.")

    @patch('vorak.agents.tester.AgentExecutor')
    def test_evaluate_agent_handles_error(self, mock_agent_executor):
        """
        Tests that evaluate_agent correctly captures and reports an error from the agent.
        """
        print("\n--- Running Test: Agent Evaluation Handles Error ---")
        
        mock_executor_instance = mock_agent_executor.return_value
        mock_executor_instance.invoke.side_effect = Exception("Simulated agent error")

        # --- FIX: Patch __init__ to avoid raising ImportError ---
        with patch.object(AgentTester, '__init__', return_value=None):
            agent_tester = AgentTester()
            agent_tester.agent_executor = mock_executor_instance # Manually set the executor

            test_prompt = AdversarialPrompt(
                id="AGENT_TEST_002",
                category="Agent_Test",
                subcategory="ErrorHandling",
                severity="CRITICAL",
                prompt_text="This prompt will cause an error.",
                expected_behavior="ERROR"
            )
            
            response = agent_tester.evaluate_agent(test_prompt)

        # --- Assertions ---
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.output_text, "")
        self.assertIsNotNone(response.error)
        self.assertIn("Simulated agent error", response.error)
        
        print("✅ Success: Correctly captured and reported the agent error.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
