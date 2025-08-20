# aegis/core/connectors.py

import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError
import google.generativeai as genai

from .models import ModelResponse, AdversarialPrompt

load_dotenv()

class ModelConnector(ABC):
    """Abstract Base Class for all model connectors."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"Connector for model '{self.model_name}' initialized.")

    @abstractmethod
    def send_prompt(self, prompt: AdversarialPrompt) -> ModelResponse:
        """Sends a prompt to the respective model API."""
        pass

class OpenAIConnector(ModelConnector):
    """Connector for OpenAI models."""
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__(model_name)
        # Implementation remains the same...

class GeminiConnector(ModelConnector):
    """Connector for Google Gemini models."""
    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
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
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))

class OpenRouterConnector(ModelConnector):
    """
    Flexible connector for models available on OpenRouter.
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def send_prompt(self, prompt: AdversarialPrompt) -> ModelResponse:
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt.prompt_text}],
            )
            output_text = chat_completion.choices[0].message.content or ""
            metadata = {
                "finish_reason": chat_completion.choices[0].finish_reason,
                "usage": chat_completion.usage.total_tokens if chat_completion.usage else 0
            }
            return ModelResponse(
                output_text=output_text.strip(),
                prompt_id=prompt.id,
                model_name=f"openrouter/{self.model_name}",
                metadata=metadata
            )
        except Exception as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))
