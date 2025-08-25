# vorak/core/connectors.py

import os
import json
import requests
import base64
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai
from PIL import Image
from io import BytesIO
from typing import List, Optional, Dict, Any

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .models import ModelResponse, AdversarialPrompt

load_dotenv()

class ModelConnector(ABC):
    """Abstract Base Class for all model connectors."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"Connector for model '{self.model_name}' initialized.")

    @abstractmethod
    def send_prompt(self, prompt: AdversarialPrompt, history: Optional[List[Dict[str, Any]]] = None) -> ModelResponse:
        """Sends a prompt to the respective model API, optionally with conversation history."""
        pass

class LocalModelConnector(ModelConnector):
    """Connector for local models using the transformers library."""
    def __init__(self, model_name: str):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("The 'transformers' and 'torch' libraries are required for local model testing. Please install them via 'pip install transformers torch'.")
        
        super().__init__(model_name)
        try:
            print(f"Loading local model from path: {self.model_name}...")
            device = 0 if torch.cuda.is_available() else -1
            print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
            
            self.pipe = pipeline(
                "text-generation", 
                model=self.model_name,
                device=device,
                torch_dtype=torch.bfloat16 if device == 0 else torch.float32
            )
            print("✅ Local model loaded successfully.")
        except Exception as e:
            raise ValueError(f"Failed to load local model '{self.model_name}'. Error: {e}")

    def send_prompt(self, prompt: AdversarialPrompt, history: Optional[List[Dict[str, Any]]] = None) -> ModelResponse:
        try:
            if prompt.image_data:
                return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error="Multi-modal prompts are not supported for local models.")

            # --- FIX: Only format as a conversation if history is provided ---
            if history:
                print("Warning: Local models do not natively support chat history. Concatenating history as context.")
                full_prompt_text = ""
                for message in history:
                    full_prompt_text += f"{message['role'].title()}: {message['content']}\n\n"
                full_prompt_text += f"User: {prompt.prompt_text}"
            else:
                full_prompt_text = prompt.prompt_text

            outputs = self.pipe(
                full_prompt_text,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
            
            output_text = outputs[0]['generated_text'] if outputs and 'generated_text' in outputs[0] else ""
            
            if output_text.startswith(full_prompt_text):
                output_text = output_text[len(full_prompt_text):].strip()

            return ModelResponse(
                output_text=output_text,
                prompt_id=prompt.id,
                model_name=f"local/{os.path.basename(self.model_name)}"
            )
        except Exception as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))


class OpenAIConnector(ModelConnector):
    """Connector for OpenAI models like GPT-4, with multi-modal support."""
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        if not api_key:
            raise ValueError("OpenAI API key was not provided.")
        self.client = OpenAI(api_key=api_key)

    def send_prompt(self, prompt: AdversarialPrompt, history: Optional[List[Dict[str, Any]]] = None) -> ModelResponse:
        try:
            messages = history or []
            
            content = [{"type": "text", "text": prompt.prompt_text}]
            if prompt.image_data:
                base64_image = base64.b64encode(prompt.image_data).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            messages.append({"role": "user", "content": content})

            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            output_text = chat_completion.choices[0].message.content or ""
            metadata = {
                "finish_reason": chat_completion.choices[0].finish_reason,
                "usage": chat_completion.usage.total_tokens if chat_completion.usage else 0
            }
            return ModelResponse(
                output_text=output_text.strip(),
                prompt_id=prompt.id,
                model_name=f"openai/{self.model_name}",
                metadata=metadata
            )
        except Exception as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))

class AnthropicConnector(ModelConnector):
    """Connector for Anthropic's Claude models, with multi-modal support."""
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        if not api_key:
            raise ValueError("Anthropic API key was not provided.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def send_prompt(self, prompt: AdversarialPrompt, history: Optional[List[Dict[str, Any]]] = None) -> ModelResponse:
        try:
            messages = history or []
            
            content = [{"type": "text", "text": prompt.prompt_text}]
            if prompt.image_data:
                img = Image.open(BytesIO(prompt.image_data))
                media_type = f"image/{img.format.lower()}"
                base64_image = base64.b64encode(prompt.image_data).decode('utf-8')
                content.insert(0, {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": base64_image},
                })
            messages.append({"role": "user", "content": content})

            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                messages=messages
            )
            output_text = message.content[0].text if message.content else ""
            metadata = {
                "stop_reason": message.stop_reason,
                "usage": message.usage.input_tokens + message.usage.output_tokens
            }
            return ModelResponse(
                output_text=output_text.strip(),
                prompt_id=prompt.id,
                model_name=f"anthropic/{self.model_name}",
                metadata=metadata
            )
        except Exception as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))

class InternalGeminiConnector(ModelConnector):
    """Connector for Google Gemini models, used internally by the AI Analyzer."""
    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        super().__init__(model_name)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY for the evaluator model is not set.")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model_name)

    def send_prompt(self, prompt: AdversarialPrompt, history: Optional[List[Dict[str, Any]]] = None) -> ModelResponse:
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
        self.client = genai.GenerativeModel(self.model_name)

    def send_prompt(self, prompt: AdversarialPrompt, history: Optional[List[Dict[str, Any]]] = None) -> ModelResponse:
        try:
            chat = self.client.start_chat(history=(history or []))
            
            content = [prompt.prompt_text]
            if prompt.image_data:
                img = Image.open(BytesIO(prompt.image_data))
                content.append(img)
            
            response = chat.send_message(content)
            output_text = response.text or ""
            metadata = {"finish_reason": response.prompt_feedback.block_reason.name if response.prompt_feedback else "UNKNOWN"}
            return ModelResponse(output_text=output_text.strip(), prompt_id=prompt.id, model_name=self.model_name, metadata=metadata)
        except Exception as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))

class OpenRouterConnector(ModelConnector):
    """Flexible connector for models on OpenRouter."""
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        if not api_key:
            raise ValueError("OpenRouter API key was not provided.")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    def send_prompt(self, prompt: AdversarialPrompt, history: Optional[List[Dict[str, Any]]] = None) -> ModelResponse:
        try:
            messages = history or []
            messages.append({"role": "user", "content": prompt.prompt_text})
            
            chat_completion = self.client.chat.completions.create(model=self.model_name, messages=messages)
            output_text = chat_completion.choices[0].message.content or ""
            metadata = {"finish_reason": chat_completion.choices[0].finish_reason, "usage": chat_completion.usage.total_tokens if chat_completion.usage else 0}
            return ModelResponse(output_text=output_text.strip(), prompt_id=prompt.id, model_name=f"openrouter/{self.model_name}", metadata=metadata)
        except Exception as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))

class CustomEndpointConnector(ModelConnector):
    """Connector for any custom REST API endpoint."""
    def __init__(self, endpoint_url: str, headers: dict):
        super().__init__(model_name=endpoint_url)
        self.endpoint_url = endpoint_url
        self.headers = headers

    def send_prompt(self, prompt: AdversarialPrompt, history: Optional[List[Dict[str, Any]]] = None) -> ModelResponse:
        try:
            payload = {"prompt": prompt.prompt_text, "history": history or []}
            response = requests.post(self.endpoint_url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            response_json = response.json()
            output_text = response_json.get("response", "Error: 'response' key not found in JSON.")
            return ModelResponse(output_text=output_text.strip(), prompt_id=prompt.id, model_name=self.model_name)
        except Exception as e:
            return ModelResponse(output_text="", prompt_id=prompt.id, model_name=self.model_name, error=str(e))
