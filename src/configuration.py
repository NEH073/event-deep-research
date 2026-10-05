import os
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""

    # Single model for most providers (simplified configuration)
    llm_model: str = Field(
        default="groq:openai/gpt-oss-20b",
        description="Primary LLM model to use for both structured output and tools",
    )

    # Optional overrides
    structured_llm_model: str | None = Field(
        default=None,
        # default="ollama:mistral-nemo:latest",
        description="Override model for structured output",
    )
    tools_llm_model: str | None = Field(
        default=None,
        # default="ollama:gpt-oss:20b",
        description="Override model for tools",
    )
    chunk_llm_model: str | None = Field(
        default=None,
        # default="ollama:gemma3:4b",
        description="Small model for chunk biographical event detection",
    )

    structured_llm_max_tokens: int = Field(
        default=4096, description="Maximum tokens for structured output model"
    )
    tools_llm_max_tokens: int = Field(
        default=4096, description="Maximum tokens for tools model"
    )

    max_structured_output_retries: int = Field(
        default=3, description="Maximum retry attempts for structured output"
    )
    max_tools_output_retries: int = Field(
        default=3, description="Maximum retry attempts for tool calls"
    )

    # Hardcoded values from graph files
    default_chunk_size: int = Field(
        default=800, description="Default chunk size for text processing"
    )
    default_overlap_size: int = Field(
        default=20, description="Default overlap size between chunks"
    )
    max_content_length: int = Field(
        default=100000, description="Maximum content length to process"
    )
    max_tool_iterations: int = Field(
        default=5, description="Maximum number of tool iterations"
    )
    max_chunks: int = Field(
        default=20,
        description="Maximum number of chunks to process for biographical event detection",
    )

    def get_llm_structured_model(self) -> str:
        """Get the LLM structured model, using overrides if provided."""
        print(f"Getting LLM structured model: {self.structured_llm_model}")
        if self.structured_llm_model:
            return self.structured_llm_model

        return self.llm_model

    def get_llm_with_tools_model(self) -> str:
        """Get the LLM with tools model, using overrides if provided."""
        print(f"Getting LLM with tools model: {self.tools_llm_model}")
        if self.tools_llm_model:
            return self.tools_llm_model

        return self.llm_model

    def get_llm_chunk_model(self) -> str:
        """Get the LLM chunk model, using overrides if provided."""
        if self.chunk_llm_model:
            return self.chunk_llm_model
        return "groq:openai/gpt-oss-20b"

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})
