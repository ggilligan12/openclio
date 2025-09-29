"""Pydantic models for structured LLM outputs"""

from pydantic import BaseModel, Field
from typing import List, Optional


class FacetAnswer(BaseModel):
    """Structured output for a single facet extraction"""
    answer: str = Field(
        description="The answer to the facet question. Clear, concise, 1-2 sentences. No PII or proper nouns."
    )


class ClusterNames(BaseModel):
    """Structured output for cluster naming"""
    names: List[str] = Field(
        description="List of cluster names, each being a clear descriptive label"
    )


class DeduplicatedNames(BaseModel):
    """Structured output for deduplicated cluster names"""
    unique_names: List[str] = Field(
        description="Deduplicated list of unique cluster names"
    )


class ClusterAssignment(BaseModel):
    """Structured output for assigning a cluster to a higher-level category"""
    assigned_category: str = Field(
        description="The name of the higher-level category this cluster belongs to"
    )


class ClusterRenaming(BaseModel):
    """Structured output for renaming a cluster based on its children"""
    summary: str = Field(
        description="A descriptive summary of what this cluster represents"
    )
    name: str = Field(
        description="A concise name/label for this cluster"
    )
