"""Abstract LLM interface to support multiple backends (VLLM, Vertex AI, etc.)"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Type

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None


class LLMInterface(ABC):
    """Abstract base class for LLM backends"""

    @abstractmethod
    def generate_batch(self, prompts: List[str], response_schema: Optional[Type] = None, **kwargs) -> List[str]:
        """
        Generate completions for a batch of prompts.

        Args:
            prompts: List of prompt strings
            response_schema: Optional Pydantic model for structured outputs (if backend supports it)
            **kwargs: Backend-specific generation parameters

        Returns:
            List of generated text completions (same length as prompts)
            If response_schema provided, returns JSON strings conforming to the schema
        """
        pass

    @abstractmethod
    def get_tokenizer(self) -> Optional[Any]:
        """
        Get the tokenizer for this LLM, if available.

        Returns:
            Tokenizer object or None if not available
        """
        pass

    @abstractmethod
    def supports_chat_template(self) -> bool:
        """
        Check if this LLM supports chat templates.

        Returns:
            True if chat templates are supported
        """
        pass
