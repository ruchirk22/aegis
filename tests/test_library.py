# tests/test_library.py

import unittest
from aegis.core.library import PromptLibrary
from aegis.core.models import AdversarialPrompt

class TestPromptLibrary(unittest.TestCase):
    """
    Unit tests for the PromptLibrary class.
    """

    @classmethod
    def setUpClass(cls):
        """Set up a PromptLibrary instance for all tests in this class."""
        cls.library = PromptLibrary() # No need to specify path anymore
        cls.library.load_prompts()

    def test_01_successful_loading(self):
        """Verify that prompts are loaded and the list is not empty."""
        print("\n--- Running Test 01: Successful Loading ---")
        all_prompts = self.library.get_all()
        self.assertIsNotNone(all_prompts)
        self.assertIsInstance(all_prompts, list)
        self.assertGreater(len(all_prompts), 0)
        self.assertIsInstance(all_prompts[0], AdversarialPrompt)
        print(f"✅ Success: Found {len(all_prompts)} prompts.")

    def test_02_filter_by_category(self):
        """Test the category filtering functionality."""
        print("\n--- Running Test 02: Filter by Category ---")
        # FIX: Use a category name that is confirmed to exist in your prompts.
        category_to_test = "Direct_Prompt_Injection"
        filtered_prompts = self.library.filter_by_category(category_to_test)
        
        self.assertGreater(len(filtered_prompts), 0)
        for prompt in filtered_prompts:
            self.assertEqual(prompt.category, category_to_test)
        print(f"✅ Success: Found {len(filtered_prompts)} prompts for category '{category_to_test}'.")

    def test_03_filter_by_severity(self):
        """Test the severity filtering functionality."""
        print("\n--- Running Test 03: Filter by Severity ---")
        severity_to_test = "HIGH"
        filtered_prompts = self.library.filter_by_severity(severity_to_test)
        
        self.assertGreater(len(filtered_prompts), 0)
        print(f"✅ Success: Found {len(filtered_prompts)} prompts with severity '{severity_to_test}'.")

        # FIX: Update the test to correctly check for CRITICAL prompts, since they now exist.
        severity_to_test_critical = "CRITICAL"
        filtered_prompts_critical = self.library.filter_by_severity(severity_to_test_critical)
        self.assertGreater(len(filtered_prompts_critical), 0)
        print(f"✅ Success: Correctly found {len(filtered_prompts_critical)} prompts with severity '{severity_to_test_critical}'.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
