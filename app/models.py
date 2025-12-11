"""Pydantic models for API requests, responses, and internal data."""
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class QuizStartRequest(BaseModel):
    """Request model for starting a quiz session."""
    email: EmailStr = Field(..., description="Student email address")
    secret: str = Field(..., min_length=1, description="Authentication secret")
    url: str = Field(..., min_length=10, description="The quiz URL to solve")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL is properly formatted."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class QuizStartResponse(BaseModel):
    """Response model for quiz start"""
    status: str = "accepted"
    message: str = "Quiz solving started in background"
    task_id: Optional[str] = None


class QuizStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class QuizResult(BaseModel):
    """Result of a quiz question"""
    url: str
    answer: Any
    correct: Optional[bool] = None
    message: Optional[str] = None
    next_url: Optional[str] = None


class PageContent(BaseModel):
    """Extracted content from a quiz page"""
    url: str
    text_content: str
    links: list[str] = []
    images: list[str] = []
    submission_endpoint: Optional[str] = None
    raw_html: Optional[str] = None


class CodeExecutionResult(BaseModel):
    """Result of code execution in sandbox"""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    timed_out: bool = False


class SubmissionPayload(BaseModel):
    """Payload sent to quiz server for answer submission"""
    email: EmailStr
    secret: str
    url: str
    answer: Any


class SubmissionResponse(BaseModel):
    """Response from quiz server after submission"""
    correct: Optional[bool] = None
    message: Optional[str] = None
    url: Optional[str] = None  # Next question URL if correct
    reason: Optional[str] = None  # Reason for wrong answer
