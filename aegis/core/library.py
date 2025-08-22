# aegis/core/library.py

import json
import os
import sys
from typing import List, Optional
from importlib import resources

# Import our data model from the models.py file
from .models import AdversarialPrompt

class PromptLibrary:
    """
    Manages the collection of adversarial prompts.
    """

    def __init__(self):
        """Initializes the PromptLibrary."""
        self.prompts: List[AdversarialPrompt] = []
        self._loaded = False

    def load_prompts(self):
        """
        Loads all prompts from .json files within the aegis.prompts package directory.
        """
        if self._loaded:
            return

        loaded_prompts = []
        # Use importlib.resources to safely access package data
        try:
            # FIX: The method for iterating over package resources changed in Python 3.10.
            # This code block handles both old and new versions.
            if sys.version_info < (3, 10):
                # For Python 3.9
                with resources.path('aegis', 'prompts') as p:
                    prompt_files = [f for f in os.listdir(p) if f.endswith('.json')]
                    for file_name in prompt_files:
                        file_path = os.path.join(p, file_name)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            prompts_data = json.load(f)
                            for prompt_dict in prompts_data:
                                prompt = AdversarialPrompt(**prompt_dict)
                                loaded_prompts.append(prompt)
            else:
                # For Python 3.10 and newer
                prompt_files_path = resources.files('aegis.prompts')
                for file_path in prompt_files_path.iterdir():
                    if file_path.is_file() and file_path.name.endswith('.json'):
                        with file_path.open('r', encoding='utf-8') as f:
                            prompts_data = json.load(f)
                            for prompt_dict in prompts_data:
                                prompt = AdversarialPrompt(**prompt_dict)
                                loaded_prompts.append(prompt)

        except (ModuleNotFoundError, FileNotFoundError):
            print("Warning: 'aegis.prompts' package not found. No prompts will be loaded.")
        except Exception as e:
            print(f"An unexpected error occurred while loading prompts: {e}")


        self.prompts = loaded_prompts
        self._loaded = True
        print(f"Successfully loaded {len(self.prompts)} prompts.")

    def _ensure_loaded(self):
        """Internal helper to load prompts on-demand."""
        if not self._loaded:
            self.load_prompts()

    def get_all(self) -> List[AdversarialPrompt]:
        """Returns all loaded prompts."""
        self._ensure_loaded()
        return self.prompts

    def filter_by_category(self, category: str) -> List[AdversarialPrompt]:
        """Filters prompts by a specific category."""
        self._ensure_loaded()
        return [p for p in self.prompts if p.category.lower() == category.lower()]

    def filter_by_severity(self, severity: str) -> List[AdversarialPrompt]:
        """Filters prompts by a specific severity level."""
        self._ensure_loaded()
        return [p for p in self.prompts if p.severity.lower() == severity.lower()]
