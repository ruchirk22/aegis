# aegis/core/connectors.py

import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# OpenAI specific imports
from openai import OpenAI, RateLimitError, APIError

# Gemini specific imports
import google.generativeai as genai

from .models import ModelResponse, AdversarialPrompt

# Load environment variables from .env file
load_dotenv()

class ModelConnector(ABC):
    """
    Abstract Base Class for all model connectors.
    
    This defines the standard interface that all concrete model connectors
    must implement, ensuring consistent behavior across different LLM providers.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"Connector for model '{self.model_name}' initialized.")

    @abstractmethod
    def send_prompt(self, prompt: AdversarialPrompt) -> ModelResponse:
        """
        Sends a prompt to the respective model API and returns a standardized response.
        
        This method must be implemented by all subclasses.
        """
        pass

class OpenAIConnector(ModelConnector):
    """
    Concrete implementation of ModelConnector for OpenAI models.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__(model_name)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.client = OpenAI(api_key=api_key)

    def send_prompt(self, prompt: AdversarialPrompt) -> ModelResponse:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt.prompt_text}],
                model=self.model_name,
            )
            
            output_text = chat_completion.choices[0].message.content or ""
            metadata = {
                "finish_reason": chat_completion.choices[0].finish_reason,
                "usage": chat_completion.usage.total_tokens if chat_completion.usage else 0
            }

            return ModelResponse(
                output_text=output_text.strip(),
                prompt_id=prompt.id,
                model_name=self.model_name,
                metadata=metadata
            )
        except RateLimitError as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=f"RateLimitError: {e}")
        except APIError as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=f"APIError: {e}")
        except Exception as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))

class GeminiConnector(ModelConnector):
    """
    Concrete implementation of ModelConnector for Google Gemini models.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        super().__init__(model_name)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model_name)

    def send_prompt(self, prompt: AdversarialPrompt) -> ModelResponse:
        try:
            response = self.client.generate_content(prompt.prompt_text)
            
            output_text = response.text or ""
            metadata = {
                "finish_reason": response.prompt_feedback.block_reason.name if response.prompt_feedback else "UNKNOWN"
            }

            return ModelResponse(
                output_text=output_text.strip(),
                prompt_id=prompt.id,
                model_name=self.model_name,
                metadata=metadata
            )
        except Exception as e:
            # The Gemini API has a more general exception type for many errors
            print(f"An unexpected error occurred for prompt ID {prompt.id}: {e}")
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))
        