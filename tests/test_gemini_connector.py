# tests/test_gemini_connector.py

import unittest
from unittest.mock import patch, MagicMock
import os

# Set a dummy API key before importing the connector
os.environ['GEMINI_API_KEY'] = 'test_key_for_unit_testing'

from vorak.core.connectors import InternalGeminiConnector
from vorak.core.models import AdversarialPrompt, ModelResponse

class TestInternalGeminiConnector(unittest.TestCase):
    """
    Unit tests for the InternalGeminiConnector class.
    
    Uses mocking to simulate the Google Generative AI API.
    """

    @patch('vorak.core.connectors.genai.GenerativeModel')
    def test_send_prompt_successful(self, MockGenerativeModel):
        """
        Verify that send_prompt correctly processes a successful Gemini API response.
        """
        print("\n--- Running Test: Successful Gemini API Call ---")

        # --- Mock Setup ---
        mock_response = MagicMock()
        mock_response.text = "This is a successful Gemini test response. "
        # Simulate the nested structure for prompt_feedback
        mock_response.prompt_feedback.block_reason.name = "SAFETY"

        mock_instance = MockGenerativeModel.return_value
        mock_instance.generate_content.return_value = mock_response

        # --- Test Execution ---
        test_prompt = AdversarialPrompt(
            id="GEMINI_TEST_001",
            category="Test",
            subcategory="Mocking",
            severity="LOW",
            prompt_text="This is a Gemini test prompt.",
            expected_behavior="RESPOND"
        )
        
        connector = InternalGeminiConnector()
        response = connector.send_prompt(test_prompt)

        # --- Assertions ---
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.output_text, "This is a successful Gemini test response.")
        self.assertEqual(response.prompt_id, "GEMINI_TEST_001")
        self.assertIsNone(response.error)
        self.assertEqual(response.metadata['finish_reason'], "SAFETY")
        
        print("✅ Success: Correctly parsed the mock Gemini API response.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
