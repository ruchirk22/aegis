# aegis/tests/test_library.py

import sys
import os
import unittest

# To import modules from the parent directory (aegis/), we add the project root to the Python path.
# This is a common pattern for making test scripts runnable from the command line.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from aegis.core.library import PromptLibrary
from aegis.core.models import AdversarialPrompt

class TestPromptLibrary(unittest.TestCase):
    """
    Unit tests for the PromptLibrary class.
    
    These tests verify the core functionalities:
    1.  Initialization of the library.
    2.  Correct loading of prompts from a JSON file.
    3.  Filtering logic for categories and severity.
    """

    @classmethod
    def setUpClass(cls):
        """Set up a PromptLibrary instance for all tests in this class."""
        # The path is relative to the project root where the test will be run
        cls.library = PromptLibrary(data_path="aegis/prompts")
        cls.library.load_prompts()

    def test_01_successful_loading(self):
        """Verify that prompts are loaded and the list is not empty."""
        print("\n--- Running Test 01: Successful Loading ---")
        all_prompts = self.library.get_all()
        self.assertIsNotNone(all_prompts)
        self.assertIsInstance(all_prompts, list)
        self.assertEqual(len(all_prompts), 2)
        self.assertIsInstance(all_prompts[0], AdversarialPrompt)
        print(f"✅ Success: Found {len(all_prompts)} prompts.")

    def test_02_filter_by_category(self):
        """Test the category filtering functionality."""
        print("\n--- Running Test 02: Filter by Category ---")
        category_to_test = "Direct_Prompt_Injection"
        filtered_prompts = self.library.filter_by_category(category_to_test)
        
        self.assertEqual(len(filtered_prompts), 2)
        # Check if all filtered prompts actually belong to the category
        for prompt in filtered_prompts:
            self.assertEqual(prompt.category, category_to_test)
        print(f"✅ Success: Found {len(filtered_prompts)} prompts for category '{category_to_test}'.")

    def test_03_filter_by_severity(self):
        """Test the severity filtering functionality."""
        print("\n--- Running Test 03: Filter by Severity ---")
        severity_to_test = "HIGH"
        filtered_prompts = self.library.filter_by_severity(severity_to_test)
        
        self.assertEqual(len(filtered_prompts), 1)
        self.assertEqual(filtered_prompts[0].id, "DPI_001")
        print(f"✅ Success: Found {len(filtered_prompts)} prompt with severity '{severity_to_test}'.")

        severity_to_test_none = "CRITICAL"
        filtered_prompts_none = self.library.filter_by_severity(severity_to_test_none)
        self.assertEqual(len(filtered_prompts_none), 0)
        print(f"✅ Success: Correctly found 0 prompts with severity '{severity_to_test_none}'.")


if __name__ == "__main__":
    # This block allows the script to be run directly.
    # unittest.main() will discover and run all tests in this file.
    print("=========================================")
    print("  Running Aegis Prompt Library Tests     ")
    print("=========================================")
    unittest.main(verbosity=2)