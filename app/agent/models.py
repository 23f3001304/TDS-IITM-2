"""Agent data models for quiz solving context and responses."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.models import PageContent, QuizResult


@dataclass
class QuizDependencies:
    """Dependencies passed to the agent tools.
    
    Attributes:
        email: Student email for identification
        secret: Authentication secret
        current_url: Current quiz question URL
        page_content: Extracted page content
        base_url: Base URL for relative URL resolution
        url_cache: Cache for fetched URL contents
    """
    email: str
    secret: str
    current_url: str
    page_content: PageContent
    base_url: str = ""
    url_cache: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize derived fields after dataclass creation."""
        parsed = urlparse(self.current_url)
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        if self.page_content and self.page_content.text_content:
            self.url_cache[self.current_url] = self.page_content.text_content
    
    def cache_url(self, url: str, content: str) -> None:
        """Add content to URL cache."""
        self.url_cache[url] = content
    
    def get_cached(self, url: str) -> Optional[str]:
        """Get cached content for URL."""
        return self.url_cache.get(url)


class QuizAnswer(BaseModel):
    """Structured output from the agent."""
    answer: Any = Field(..., description="The computed answer")
    submission_url: str = Field(..., description="URL to submit the answer")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(default="", description="Explanation of the answer")


@dataclass
class QuizContext:
    """Context for a quiz solving session.
    
    Tracks state across multiple questions in a quiz chain.
    """
    email: str
    secret: str
    current_url: str
    attempt_number: int = 0
    results: List[QuizResult] = field(default_factory=list)
    last_failure_reason: str = ""
    last_wrong_answer: str = ""
    
    def record_failure(self, answer: str, reason: str) -> None:
        """Record a failed attempt."""
        self.last_wrong_answer = answer
        self.last_failure_reason = reason
        self.attempt_number += 1
    
    def reset_for_next_question(self, next_url: str) -> None:
        """Reset state for next question."""
        self.current_url = next_url
        self.attempt_number = 0
        self.last_failure_reason = ""
        self.last_wrong_answer = ""
