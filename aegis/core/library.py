# aegis/core/library.py

import json
import os
from typing import List, Optional

# Import our data model from the models.py file
from .models import AdversarialPrompt

class PromptLibrary:
    """
    Manages the collection of adversarial prompts.

    This class is responsible for loading prompts from a specified directory,
    validating their structure, and providing methods to filter and access them.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initializes the PromptLibrary.

        It now intelligently finds the 'prompts' directory relative to this file's
        location, making the application runnable from anywhere on the system.

        Args:
            data_path (str, optional): An optional override for the prompt data path.
        """
        if data_path:
            self.data_path = data_path
        else:
            # --- Flawless Path Resolution ---
            # Get the directory where this library.py file is located.
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct a robust, absolute path to the 'prompts' directory.
            self.data_path = os.path.join(base_dir, '..', 'prompts')

        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"The specified data path does not exist: {self.data_path}")
            
        self.prompts: List[AdversarialPrompt] = []
        self._loaded = False

    def load_prompts(self):
        """
        Loads all prompts from .json files in the data_path directory.
        """
        if self._loaded:
            return

        loaded_prompts = []
        for filename in os.listdir(self.data_path):
            if filename.endswith(".json"):
                file_path = os.path.join(self.data_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        prompts_data = json.load(f)
                        for prompt_dict in prompts_data:
                            prompt = AdversarialPrompt(**prompt_dict)
                            loaded_prompts.append(prompt)
                except Exception as e:
                    print(f"An unexpected error occurred while loading {filename}: {e}")
        
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