"""
models.py — shared data structures.
"""

from dataclasses import dataclass, field


@dataclass
class ModelSuggestion:
    name: str
    url: str
    score: float
    is_free: bool
    has_provider: bool
    provider: str | None
    supported_task: str | None = None
    inferred_method: str | None = None
    has_chat_template: bool = False
    pipeline_tag: str | None = None
    tags: list = field(default_factory=list)
    gated: bool = False