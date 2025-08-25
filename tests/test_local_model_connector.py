# tests/test_local_model_connector.py

import unittest
from unittest.mock import patch, MagicMock
import os

# Set a dummy API key to satisfy other connector imports if needed
os.environ['GEMINI_API_KEY'] = 'test_key_for_unit_testing'

# We need to import the class we're testing
from vorak.core.connectors import LocalModelConnector
from vorak.core.models import AdversarialPrompt, ModelResponse

# We don't need to mock the entire library, just the 'pipeline' function
# where it's used inside the connectors module.

class TestLocalModelConnector(unittest.TestCase):
    """
    Unit tests for the LocalModelConnector class.
    
    Uses mocking to simulate the Hugging Face transformers pipeline.
    """

    # Patch the pipeline function directly in the connectors module
    @patch('vorak.core.connectors.pipeline')
    def test_send_prompt_successful(self, mock_pipeline_func):
        """
        Verify that send_prompt correctly processes a successful local model response.
        """
        print("\n--- Running Test: Successful Local Model Call ---")

        # --- Mock Setup ---
        # The prompt text we will send
        prompt_text = "This is a local test prompt."
        
        # The full output from the mocked pipeline, including the prompt
        pipeline_output = prompt_text + " This is a successful local model response."
        
        # Configure the mock pipeline object that the function will return
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.return_value = [{'generated_text': pipeline_output}]
        
        # Make the 'pipeline' function return our mock instance
        mock_pipeline_func.return_value = mock_pipe_instance

        # --- Test Execution ---
        test_prompt = AdversarialPrompt(
            id="LOCAL_TEST_001",
            category="Test",
            subcategory="LocalMocking",
            severity="LOW",
            prompt_text=prompt_text,
            expected_behavior="RESPOND"
        )
        
        # Instantiate the connector. The 'pipeline' function inside its __init__ will be mocked.
        connector = LocalModelConnector(model_name="/fake/path/to/model")
        response = connector.send_prompt(test_prompt)

        # --- Assertions ---
        self.assertIsInstance(response, ModelResponse)
        
        # The connector should have stripped the original prompt from the output
        self.assertEqual(response.output_text, "This is a successful local model response.")
        
        self.assertEqual(response.prompt_id, "LOCAL_TEST_001")
        self.assertIsNone(response.error)
        self.assertEqual(response.model_name, "local/model")
        
        # Verify that the pipeline instance was called with the correct text
        mock_pipe_instance.assert_called_once_with(
            prompt_text,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        print("✅ Success: Correctly parsed the mock local model response.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
