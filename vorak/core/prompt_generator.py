# vorak/core/prompt_generator.py

import nlpaug.augmenter.word as naw
import random
from typing import List

from .models import AdversarialPrompt
from .prompt_manager import PromptManager

class PromptGenerator:
    """
    Generates new adversarial prompts by augmenting existing ones.
    """

    def __init__(self):
        """Initializes the PromptGenerator."""
        print("Initializing prompt generator...")
        self.prompt_manager = PromptManager()
        # Using a pre-trained WordNet model for synonym augmentation.
        # This will download the model on first run.
        self.synonym_augmenter = naw.SynonymAug(aug_src='wordnet')
        print("Prompt generator initialized successfully.")

    def _generate_unique_id(self, base_prompt: AdversarialPrompt, existing_ids: set) -> str:
        """Generates a new, unique ID for a generated prompt."""
        base_id_parts = base_prompt.id.split('_')
        prefix = f"GEN_{base_id_parts[0]}"
        
        i = 1
        while True:
            new_id = f"{prefix}_{i:03d}"
            if new_id not in existing_ids:
                return new_id
            i += 1

    def generate_prompts(
        self,
        base_category: str,
        num_to_generate: int,
        strategy: str = 'synonym'
    ) -> List[AdversarialPrompt]:
        """
        Generates a specified number of new prompts based on a category.

        Args:
            base_category (str): The category to source base prompts from.
            num_to_generate (int): The number of new prompts to create.
            strategy (str): The augmentation strategy to use ('synonym' is currently supported).

        Returns:
            List[AdversarialPrompt]: A list of newly generated adversarial prompts.
        """
        self.prompt_manager._ensure_loaded()
        base_prompts = self.prompt_manager.filter_by_category(base_category)
        
        if not base_prompts:
            print(f"Error: No prompts found in category '{base_category}' to use as a base.")
            return []

        if strategy != 'synonym':
            print(f"Warning: Strategy '{strategy}' is not yet supported. Defaulting to 'synonym'.")
            # We will add more strategies like 'llm' in future phases.

        generated_prompts = []
        # Get all existing IDs to prevent collisions
        existing_ids = {p.id for p in self.prompt_manager.get_all()}

        print(f"Generating {num_to_generate} new prompts using '{strategy}' strategy...")
        for i in range(num_to_generate):
            # Choose a random prompt from the base category to augment
            source_prompt = random.choice(base_prompts)
            
            # Augment the text
            augmented_text = self.synonym_augmenter.augment(source_prompt.prompt_text)
            
            # Ensure the augmented text is a single string if the library returns a list
            if isinstance(augmented_text, list):
                augmented_text = augmented_text[0]

            new_id = self._generate_unique_id(source_prompt, existing_ids)
            existing_ids.add(new_id) # Add to the set to avoid duplicates in the same run

            new_prompt = AdversarialPrompt(
                id=new_id,
                category=source_prompt.category,
                subcategory=f"Generated_{strategy}",
                severity=source_prompt.severity,
                prompt_text=augmented_text,
                expected_behavior=source_prompt.expected_behavior,
                tags=source_prompt.tags + ["generated"]
            )
            generated_prompts.append(new_prompt)
        
        print(f"Successfully generated {len(generated_prompts)} prompts.")
        return generated_prompts

