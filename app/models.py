from pydantic import BaseModel, EmailStr, HttpUrl, Field
from typing import Optional, Any
from enum import Enum


class QuizStartRequest(BaseModel):
    """Request model for starting a quiz session"""
    email: EmailStr
    secret: str
    url: str = Field(..., description="The quiz URL to solve")


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
