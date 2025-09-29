"""Vertex AI LLM implementation"""

import time
import json
from typing import List, Optional, Any, Dict, Type
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from pydantic import BaseModel

from .llm_interface import LLMInterface


class VertexLLMInterface(LLMInterface):
    """
    Vertex AI implementation of LLM interface.

    Designed for Google Colab where authentication is handled automatically
    via workload identity.
    """

    def __init__(
        self,
        model_name: str,
        project_id: str,
        location: str = "us-central1",
        max_retries: int = 3,
        requests_per_minute: int = 60,
        max_output_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 40,
    ):
        """
        Initialize Vertex AI LLM interface.

        Args:
            model_name: Vertex AI model name (e.g., "gemini-1.5-pro", "gemini-1.5-flash")
            project_id: GCP project ID
            location: GCP region (default: us-central1)
            max_retries: Maximum retry attempts for failed requests
            requests_per_minute: Rate limit for API calls
            max_output_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        """
        self.model_name = model_name
        self.project_id = project_id
        self.location = location
        self.max_retries = max_retries
        self.requests_per_minute = requests_per_minute
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        # Rate limiting
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0

        # Initialize Vertex AI (lazy import so it only fails if actually used)
        self._init_vertex()

    def _init_vertex(self):
        """Initialize Vertex AI client"""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel, GenerationConfig

            vertexai.init(project=self.project_id, location=self.location)

            self.generation_config = GenerationConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

            self.model = GenerativeModel(self.model_name)

        except ImportError:
            raise ImportError(
                "Vertex AI dependencies not installed. "
                "Install with: pip install google-cloud-aiplatform"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Vertex AI. "
                f"Make sure project_id '{self.project_id}' is correct and "
                f"Vertex AI API is enabled. Error: {e}"
            )

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        self.last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    def _generate_single(self, prompt: str, response_schema: Optional[Type[BaseModel]] = None) -> str:
        """Generate completion for a single prompt with retries"""
        self._rate_limit()

        try:
            # Use structured output if schema provided
            if response_schema is not None:
                config = {
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                }
                from vertexai.generative_models import GenerationConfig
                generation_config = GenerationConfig(**config)

                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
            else:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                )

            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError(f"Unexpected response format: {response}")

        except Exception as e:
            # Add context to error
            error_msg = str(e)
            if "quota" in error_msg.lower():
                raise Exception(
                    f"Vertex AI quota exceeded. "
                    f"Check your project quota at: "
                    f"https://console.cloud.google.com/iam-admin/quotas?project={self.project_id}"
                ) from e
            elif "permission" in error_msg.lower():
                raise Exception(
                    f"Permission denied. Make sure Vertex AI API is enabled and "
                    f"you have the necessary permissions."
                ) from e
            else:
                raise

    def generate_batch(self, prompts: List[str], response_schema: Optional[Type[BaseModel]] = None, **kwargs) -> List[str]:
        """
        Generate completions for a batch of prompts.

        Args:
            prompts: List of prompt strings
            response_schema: Optional Pydantic model for structured outputs
            **kwargs: Additional generation parameters (override defaults)

        Returns:
            List of generated text completions (JSON strings if response_schema provided)
        """
        # Override generation config if kwargs provided
        if kwargs:
            from vertexai.generative_models import GenerationConfig
            self.generation_config = GenerationConfig(
                max_output_tokens=kwargs.get('max_tokens', self.max_output_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                top_p=kwargs.get('top_p', self.top_p),
                top_k=kwargs.get('top_k', self.top_k),
            )

        results = []
        for prompt in tqdm(prompts, desc="Generating with Vertex AI"):
            result = self._generate_single(prompt, response_schema=response_schema)
            results.append(result)

        return results

    def get_tokenizer(self) -> Optional[Any]:
        """Vertex AI doesn't expose a tokenizer"""
        return None

    def supports_chat_template(self) -> bool:
        """Vertex AI doesn't use chat templates in the same way"""
        return False
