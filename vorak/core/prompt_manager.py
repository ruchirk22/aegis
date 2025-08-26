# vorak/core/prompt_manager.py

import json
import os
from typing import List, Optional
from importlib import resources
import sys

# Import our data model from the models.py file
from .models import AdversarialPrompt

class PromptManager:
    """
    Manages the collection of adversarial prompts from a central library file.
    Provides functionality to load, filter, and add new prompts.
    """

    def __init__(self, library_file: str = "prompt_library.json"):
        """
        Initializes the PromptManager.

        Args:
            library_file (str): The name of the JSON file in the vorak.prompts package.
        """
        self.prompts: List[AdversarialPrompt] = []
        self.library_file = library_file
        self._loaded = False
        self._library_path = self._get_library_path()

    def _get_library_path(self) -> Optional[str]:
        """Determines the full path to the prompt library file."""
        try:
            # This approach is robust for both installed packages and local development
            if sys.version_info < (3, 9):
                 with resources.path('vorak.prompts', self.library_file) as p:
                     return str(p)
            else:
                return str(resources.files('vorak.prompts').joinpath(self.library_file))
        except (ModuleNotFoundError, FileNotFoundError):
            print(f"Warning: Could not find the prompt library '{self.library_file}'.")
            return None

    def load_prompts(self):
        """
        Loads all prompts from the consolidated prompt_library.json file.
        """
        if self._loaded:
            return

        if not self._library_path or not os.path.exists(self._library_path):
            print("Error: Prompt library path not found. Cannot load prompts.")
            self.prompts = []
            self._loaded = True
            return

        try:
            with open(self._library_path, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)
                self.prompts = [AdversarialPrompt(**p) for p in prompts_data]
            print(f"Successfully loaded {len(self.prompts)} prompts from '{self.library_file}'.")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error loading or parsing prompts from '{self._library_path}': {e}")
            self.prompts = []

        self._loaded = True

    def _ensure_loaded(self):
        """Internal helper to load prompts on-demand."""
        if not self._loaded:
            self.load_prompts()

    def get_all(self) -> List[AdversarialPrompt]:
        """Returns all loaded prompts."""
        self._ensure_loaded()
        return self.prompts

    def id_exists(self, prompt_id: str) -> bool:
        """Checks if a prompt with the given ID already exists."""
        self._ensure_loaded()
        return any(p.id == prompt_id for p in self.prompts)

    def filter_by_category(self, category: str) -> List[AdversarialPrompt]:
        """Filters prompts by a specific category (case-insensitive)."""
        self._ensure_loaded()
        return [p for p in self.prompts if p.category.lower() == category.lower()]

    def filter_by_severity(self, severity: str) -> List[AdversarialPrompt]:
        """Filters prompts by a specific severity level (case-insensitive)."""
        self._ensure_loaded()
        return [p for p in self.prompts if p.severity.lower() == severity.lower()]

    def add_prompt(self, prompt: AdversarialPrompt, save: bool = True) -> bool:
        """
        Adds a new prompt to the library and optionally saves it back to the file.
        """
        self._ensure_loaded()
        if self.id_exists(prompt.id):
            print(f"Error: Prompt with ID '{prompt.id}' already exists.")
            return False

        self.prompts.append(prompt)
        if save:
            self.save_library()
        return True

    def save_library(self):
        """Saves the current list of prompts back to the JSON file."""
        if not self._library_path:
            print("Error: Library path not set. Cannot save prompts.")
            return

        try:
            with open(self._library_path, 'w', encoding='utf-8') as f:
                # Convert list of dataclasses to list of dicts for JSON serialization
                prompts_as_dicts = [p.to_dict() for p in self.prompts]
                json.dump(prompts_as_dicts, f, indent=2)
            print(f"Successfully saved {len(self.prompts)} prompts to '{self._library_path}'.")
        except IOError as e:
            print(f"Error writing to prompt library file '{self._library_path}': {e}")
