# SENTR-FRAMEWORK/tests/test_agent_tester.py

import unittest
from unittest.mock import patch, MagicMock
import os

# --- CHANGE: Set GEMINI_API_KEY for testing ---
os.environ['GEMINI_API_KEY'] = 'test_gemini_key'
os.environ['TAVILY_API_KEY'] = 'test_tavily_key'

from sentr.agents.tester import AgentTester
from sentr.core.models import AdversarialPrompt, ModelResponse

class TestAgentTester(unittest.TestCase):
    """
    Unit tests for the AgentTester class.
    """

    # --- CHANGE: Update mocks for Gemini ---
    @patch('sentr.agents.tester.AgentExecutor')
    @patch('sentr.agents.tester.ChatGoogleGenerativeAI')
    @patch('sentr.agents.tester.TavilySearchResults')
    @patch('sentr.agents.tester.ChatPromptTemplate')
    @patch('sentr.agents.tester.create_tool_calling_agent')
    def test_evaluate_agent_successful(self, mock_create_agent, mock_prompt_template, mock_tavily, mock_chat_gemini, mock_agent_executor):
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
        
        with patch.object(AgentTester, '_create_sample_agent', return_value=mock_executor_instance):
            agent_tester = AgentTester()
            
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

    @patch('sentr.agents.tester.AgentExecutor')
    def test_evaluate_agent_handles_error(self, mock_agent_executor):
        """
        Tests that evaluate_agent correctly captures and reports an error from the agent.
        """
        print("\n--- Running Test: Agent Evaluation Handles Error ---")
        
        mock_executor_instance = mock_agent_executor.return_value
        mock_executor_instance.invoke.side_effect = Exception("Simulated agent error")

        with patch.object(AgentTester, '_create_sample_agent', return_value=mock_executor_instance):
            agent_tester = AgentTester()
            
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