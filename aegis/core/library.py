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

    def __init__(self, data_path: str = "aegis/prompts"):
        """
        Initializes the PromptLibrary.

        Args:
            data_path (str): The path to the directory containing prompt JSON files.
        """
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"The specified data path does not exist: {data_path}")
        self.data_path = data_path
        self.prompts: List[AdversarialPrompt] = []
        self._loaded = False

    def load_prompts(self):
        """
        Loads all prompts from .json files in the data_path directory.

        It iterates through each file, validates the content against the
        AdversarialPrompt model, and populates the self.prompts list.
        """
        if self._loaded:
            print("Prompts have already been loaded.")
            return

        loaded_prompts = []
        for filename in os.listdir(self.data_path):
            if filename.endswith(".json"):
                file_path = os.path.join(self.data_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        prompts_data = json.load(f)
                        for prompt_dict in prompts_data:
                            # Use dictionary unpacking to create AdversarialPrompt instances
                            # This is robust and handles missing optional fields gracefully
                            prompt = AdversarialPrompt(**prompt_dict)
                            loaded_prompts.append(prompt)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {filename}.")
                except TypeError as e:
                    print(f"Warning: Mismatch between JSON and AdversarialPrompt model in {filename}: {e}")
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

    def add_custom_prompt(self, prompt: AdversarialPrompt) -> bool:
        """
        Adds a custom, user-defined prompt to the library in memory.
        (Implementation to be expanded in a future phase).
        """
        self._ensure_loaded()
        # Basic validation to prevent duplicates
        if any(p.id == prompt.id for p in self.prompts):
            print(f"Error: Prompt with ID '{prompt.id}' already exists.")
            return False
        self.prompts.append(prompt)
        return True

    def export_prompts(self, format: str) -> Optional[str]:
        """
        Exports the current library to a specified format.
        (Implementation to be expanded in a future phase).
        """
        self._ensure_loaded()
        if format.lower() == 'json':
            return json.dumps([p.to_dict() for p in self.prompts], indent=2)
        else:
            print(f"Error: Export format '{format}' not supported yet.")
            return None
