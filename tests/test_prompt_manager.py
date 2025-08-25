# tests/test_prompt_manager.py

import unittest
import os
import json
from unittest.mock import patch, mock_open

from vorak.core.prompt_manager import PromptManager
from vorak.core.models import AdversarialPrompt

# A sample list of prompts to be used for mocking the JSON file
SAMPLE_PROMPTS_DATA = [
    {
        "id": "TEST_001",
        "category": "Test_Category_1",
        "subcategory": "Sub1",
        "severity": "HIGH",
        "prompt_text": "This is the first test prompt.",
        "expected_behavior": "REJECT"
    },
    {
        "id": "TEST_002",
        "category": "Test_Category_1",
        "subcategory": "Sub2",
        "severity": "MEDIUM",
        "prompt_text": "This is the second test prompt.",
        "expected_behavior": "REJECT"
    },
    {
        "id": "TEST_003",
        "category": "Test_Category_2",
        "subcategory": "Sub1",
        "severity": "HIGH",
        "prompt_text": "This is the third test prompt.",
        "expected_behavior": "REJECT"
    }
]

class TestPromptManager(unittest.TestCase):
    """
    Unit tests for the new PromptManager class.
    """

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps(SAMPLE_PROMPTS_DATA))
    @patch("os.path.exists", return_value=True)
    def setUp(self, mock_exists, mock_file):
        """Set up a fresh PromptManager instance for each test."""
        # We patch 'open' and 'os.path.exists' to avoid actual file I/O
        self.manager = PromptManager()
        # The manager loads on first access, so we call get_all() to trigger it
        self.manager.get_all()

    def test_01_successful_loading(self):
        """Verify that prompts are loaded correctly from the mock JSON data."""
        print("\n--- Running Test 01: Successful Loading ---")
        all_prompts = self.manager.get_all()
        self.assertEqual(len(all_prompts), 3)
        self.assertIsInstance(all_prompts[0], AdversarialPrompt)
        self.assertEqual(all_prompts[0].id, "TEST_001")
        print(f"✅ Success: Correctly loaded {len(all_prompts)} prompts.")

    def test_02_filter_by_category(self):
        """Test the category filtering functionality."""
        print("\n--- Running Test 02: Filter by Category ---")
        filtered = self.manager.filter_by_category("Test_Category_1")
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(p.category == "Test_Category_1" for p in filtered))

        # Test case-insensitivity
        filtered_lower = self.manager.filter_by_category("test_category_1")
        self.assertEqual(len(filtered_lower), 2)
        print(f"✅ Success: Correctly filtered to {len(filtered)} prompts.")

    def test_03_filter_by_severity(self):
        """Test the severity filtering functionality."""
        print("\n--- Running Test 03: Filter by Severity ---")
        filtered = self.manager.filter_by_severity("HIGH")
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(p.severity == "HIGH" for p in filtered))
        print(f"✅ Success: Correctly filtered to {len(filtered)} prompts.")

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps(SAMPLE_PROMPTS_DATA))
    @patch("os.path.exists", return_value=True)
    def test_04_add_prompt_and_save(self, mock_exists, mock_file):
        """Test adding a new prompt and saving the library."""
        print("\n--- Running Test 04: Add and Save Prompt ---")
        manager = PromptManager() # Create a new instance for this test
        manager.load_prompts()

        new_prompt = AdversarialPrompt(
            id="TEST_004", category="Test_Category_2", subcategory="Sub3",
            severity="LOW", prompt_text="A new prompt.", expected_behavior="REJECT"
        )

        # Add the prompt
        result = manager.add_prompt(new_prompt, save=True)
        self.assertTrue(result)
        self.assertEqual(len(manager.get_all()), 4)

        # Verify that the 'save_library' method called 'open' to write the file
        mock_file.assert_called_with(manager._library_path, 'w', encoding='utf-8')

        # FIX: A more robust way to capture the written content from the mock file handle.
        # This joins all arguments from all 'write' calls, in case json.dump writes in chunks.
        handle = mock_file()
        all_written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        written_data = json.loads(all_written_content)

        self.assertEqual(len(written_data), 4)
        self.assertEqual(written_data[-1]["id"], "TEST_004")
        print("✅ Success: Prompt added and library saved correctly.")

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps(SAMPLE_PROMPTS_DATA))
    @patch("os.path.exists", return_value=True)
    def test_05_add_duplicate_prompt_fails(self, mock_exists, mock_file):
        """Test that adding a prompt with a duplicate ID fails."""
        print("\n--- Running Test 05: Duplicate Prompt ID Fails ---")
        duplicate_prompt = AdversarialPrompt(
            id="TEST_001", category="New", subcategory="New",
            severity="LOW", prompt_text="Duplicate.", expected_behavior="REJECT"
        )
        result = self.manager.add_prompt(duplicate_prompt, save=False)
        self.assertFalse(result)
        # Ensure the prompt list was not modified
        self.assertEqual(len(self.manager.get_all()), 3)
        print("✅ Success: Correctly prevented adding a duplicate prompt.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
