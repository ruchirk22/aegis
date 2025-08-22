# aegis/core/connectors.py

import os
import json
import requests
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from PIL import Image
from io import BytesIO

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

class InternalGeminiConnector(ModelConnector):
    """
    Connector for Google Gemini models, used internally by the AI Analyzer.
    (This remains text-only for now as the internal analyzer is text-based).
    """
    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        super().__init__(model_name)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY for the evaluator model is not set in the environment.")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model_name)

    def send_prompt(self, prompt: AdversarialPrompt) -> ModelResponse:
        try:
            response = self.client.generate_content(prompt.prompt_text)
            output_text = response.text or ""
            metadata = {"finish_reason": response.prompt_feedback.block_reason.name if response.prompt_feedback else "UNKNOWN"}
            return ModelResponse(output_text=output_text.strip(), prompt_id=prompt.id, model_name=self.model_name, metadata=metadata)
        except Exception as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))

class UserProvidedGeminiConnector(ModelConnector):
    """Connector for Google Gemini models using a user-provided API key, with multi-modal support."""
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        if not api_key:
            raise ValueError("Gemini API key was not provided.")
        genai.configure(api_key=api_key)
        # Ensure we use a model that supports vision
        if "vision" not in model_name and "flash" not in model_name:
             print(f"Warning: Model '{model_name}' may not support vision. Using it anyway.")
        self.client = genai.GenerativeModel(self.model_name)

    def send_prompt(self, prompt: AdversarialPrompt) -> ModelResponse:
        try:
            content = [prompt.prompt_text]
            if prompt.image_data:
                # If image data is present, open it with PIL and add to the content
                img = Image.open(BytesIO(prompt.image_data))
                content.append(img)
            
            response = self.client.generate_content(content)
            output_text = response.text or ""
            metadata = {"finish_reason": response.prompt_feedback.block_reason.name if response.prompt_feedback else "UNKNOWN"}
            return ModelResponse(output_text=output_text.strip(), prompt_id=prompt.id, model_name=self.model_name, metadata=metadata)
        except Exception as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))

class OpenRouterConnector(ModelConnector):
    """Flexible connector for models on OpenRouter. (Currently text-only)."""
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        if not api_key:
            raise ValueError("OpenRouter API key was not provided.")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    def send_prompt(self, prompt: AdversarialPrompt) -> ModelResponse:
        try:
            # Note: This connector would need significant changes for multi-modal
            chat_completion = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt.prompt_text}])
            output_text = chat_completion.choices[0].message.content or ""
            metadata = {"finish_reason": chat_completion.choices[0].finish_reason, "usage": chat_completion.usage.total_tokens if chat_completion.usage else 0}
            return ModelResponse(output_text=output_text.strip(), prompt_id=prompt.id, model_name=f"openrouter/{self.model_name}", metadata=metadata)
        except Exception as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))

class CustomEndpointConnector(ModelConnector):
    """Connector for any custom REST API endpoint. (Currently text-only)."""
    def __init__(self, endpoint_url: str, headers: dict):
        super().__init__(model_name=endpoint_url)
        self.endpoint_url = endpoint_url
        self.headers = headers

    def send_prompt(self, prompt: AdversarialPrompt) -> ModelResponse:
        try:
            payload = {"prompt": prompt.prompt_text}
            response = requests.post(self.endpoint_url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            response_json = response.json()
            output_text = response_json.get("response", "Error: 'response' key not found in JSON.")
            return ModelResponse(output_text=output_text.strip(), prompt_id=prompt.id, model_name=self.model_name)
        except Exception as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))
