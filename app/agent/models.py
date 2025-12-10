"""
Agent Data Models
"""
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from app.models import PageContent


@dataclass
class QuizDependencies:
    """Dependencies passed to the agent tools"""
    email: str
    secret: str
    current_url: str
    page_content: PageContent
    base_url: str = ""
    url_cache: dict = field(default_factory=dict)

    def __post_init__(self):
        parsed = urlparse(self.current_url)
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        if self.page_content and self.page_content.text_content:
            self.url_cache[self.current_url] = self.page_content.text_content


class QuizAnswer(BaseModel):
    """Structured output from the agent"""
    answer: Any
    submission_url: str
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class QuizContext:
    """Context for a quiz solving session"""
    email: str
    secret: str
    current_url: str
    attempt_number: int = 0
    results: list = field(default_factory=list)
    last_failure_reason: str = ""
    last_wrong_answer: str = ""
