# aegis/tests/test_gemini_connector.py

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Set a dummy API key for testing purposes
os.environ['GEMINI_API_KEY'] = 'test_key_for_unit_testing'

from aegis.core.connectors import GeminiConnector
from aegis.core.models import AdversarialPrompt, ModelResponse

class TestGeminiConnector(unittest.TestCase):
    """
    Unit tests for the GeminiConnector class.
    
    Uses mocking to simulate the Google Generative AI API.
    """

    @patch('aegis.core.connectors.genai.GenerativeModel')
    def test_01_send_prompt_successful(self, MockGenerativeModel):
        """
        Verify that send_prompt correctly processes a successful Gemini API response.
        """
        print("\n--- Running Test 01: Successful Gemini API Call ---")

        # --- Mock Setup ---
        # Create a mock object that simulates the structure of the Gemini API response.
        mock_response = MagicMock()
        mock_response.text = "This is a successful Gemini test response. "
        mock_response.prompt_feedback.block_reason.name = "SAFETY"

        # Configure the mock Gemini client to return our mock response object
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
        
        # Initialize the connector which will use the mocked client
        connector = GeminiConnector()
        response = connector.send_prompt(test_prompt)

        # --- Assertions ---
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.output_text, "This is a successful Gemini test response.")
        self.assertEqual(response.prompt_id, "GEMINI_TEST_001")
        self.assertIsNone(response.error)
        self.assertEqual(response.metadata['finish_reason'], "SAFETY")
        
        print("✅ Success: Correctly parsed the mock Gemini API response.")


if __name__ == "__main__":
    print("=========================================")
    print("   Running Aegis Gemini Connector Tests   ")
    print("=========================================")
    unittest.main(verbosity=2)
