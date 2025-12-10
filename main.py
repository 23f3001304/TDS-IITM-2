"""
Autonomous Analyst - Quiz Solving API Server
Main FastAPI application entry point
"""
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError
from loguru import logger
import json
import sys

from app.config import settings
from app.models import QuizStartRequest, QuizStartResponse, QuizResult
from app.agent import QuizAgent, QuizContext
from app.vision import vision
from app.sandbox import sandbox
from app.action import action


# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Store for tracking background tasks
task_store: Dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup/shutdown"""
    logger.info("Autonomous Analyst starting up...")
    logger.info(f"Server running on {settings.host}:{settings.port}")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down...")
    vision.shutdown()
    sandbox.shutdown()
    await action.close()


app = FastAPI(
    title="Autonomous Analyst",
    description="AI-powered quiz solving API using ReAct pattern",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handlers for proper HTTP status codes
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return HTTP 400 for invalid JSON or validation errors"""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Invalid request format", "errors": exc.errors()}
    )


@app.exception_handler(json.JSONDecodeError)
async def json_exception_handler(request: Request, exc: json.JSONDecodeError):
    """Return HTTP 400 for invalid JSON"""
    logger.warning(f"JSON decode error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Invalid JSON"}
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions properly"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


async def solve_quiz_task(task_id: str, email: str, secret: str, url: str):
    """Background task to solve the quiz"""
    logger.info(f"Task {task_id}: Starting quiz solving...")
    
    task_store[task_id]["status"] = "running"
    
    try:
        context = QuizContext(
            email=email,
            secret=secret,
            current_url=url
        )
        
        agent = QuizAgent()
        results = await agent.solve_quiz(context)
        
        task_store[task_id]["status"] = "completed"
        task_store[task_id]["results"] = [r.model_dump() for r in results]
        
        logger.info(f"Task {task_id}: Completed with {len(results)} results")
        
    except Exception as e:
        logger.error(f"Task {task_id}: Failed with error: {e}")
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["error"] = str(e)
    
    finally:
        # Cleanup sandbox
        sandbox.cleanup()


@app.post("/start", status_code=status.HTTP_200_OK)
async def start_quiz(request: QuizStartRequest, background_tasks: BackgroundTasks):
    """
    Start a quiz solving session.
    
    - Returns HTTP 200 JSON if secret matches (starts solving in background)
    - Returns HTTP 400 for invalid JSON
    - Returns HTTP 403 for invalid secret
    """
    # Validate secret - return 403 if invalid
    if request.secret != settings.secret_key:
        logger.warning(f"Invalid secret from {request.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid secret"
        )
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Store task info
    task_store[task_id] = {
        "status": "pending",
        "email": request.email,
        "url": request.url,
        "started_at": time.time(),
        "results": None,
        "error": None
    }
    
    # Spawn background task to solve quiz
    background_tasks.add_task(
        solve_quiz_task,
        task_id,
        request.email,
        request.secret,
        request.url
    )
    
    logger.info(f"Quiz task {task_id} started for {request.email} - URL: {request.url}")
    
    # Return 200 OK immediately
    return {
        "status": "success",
        "message": "Secret verified. Quiz solving started.",
        "task_id": task_id
    }


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a quiz solving task"""
    if task_id not in task_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    return task_store[task_id]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@app.post("/solve-sync")
async def solve_quiz_sync(request: QuizStartRequest):
    """
    Synchronous quiz solving endpoint.
    Blocks until the quiz is solved or times out.
    Useful for testing.
    """
    # Simple auth check
    if request.secret != settings.secret_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid secret"
        )
    
    try:
        context = QuizContext(
            email=request.email,
            secret=request.secret,
            current_url=request.url
        )
        
        agent = QuizAgent()
        results = await agent.solve_quiz(context)
        
        return {
            "status": "completed",
            "results": [r.model_dump() for r in results]
        }
        
    except Exception as e:
        logger.error(f"Sync solve failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        sandbox.cleanup()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
