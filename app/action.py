"""Action Module - Handles submission of answers to the quiz server."""
from typing import Any, Final, Optional

import httpx
from loguru import logger

from app.models import SubmissionPayload, SubmissionResponse


# Constants
SUBMISSION_TIMEOUT: Final[int] = 60
CONTENT_TYPE_JSON: Final[str] = "application/json"


class ActionModule:
    """
    Handles communication with the quiz server for answer submission.
    
    Uses connection pooling for efficient HTTP requests.
    """
    
    __slots__ = ('client',)
    
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=SUBMISSION_TIMEOUT, 
            follow_redirects=True
        )
    
    async def submit_answer(
        self,
        endpoint: str,
        email: str,
        secret: str,
        url: str,
        answer: Any
    ) -> SubmissionResponse:
        """
        Submit an answer to the quiz server.
        
        Args:
            endpoint: The submission endpoint URL
            email: Student email
            secret: Authentication secret
            url: The quiz question URL
            answer: The computed answer
            
        Returns:
            SubmissionResponse with result
        """
        payload = {
            "email": email,
            "secret": secret,
            "url": url,
            "answer": answer
        }
        
        logger.info(f"Submitting answer to {endpoint}")
        logger.debug(f"Payload: {payload}")
        
        try:
            response = await self.client.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            logger.info(f"Response text (raw): {response.text}")
            
            try:
                data = response.json()
                logger.info("=" * 60)
                logger.info("===== SUBMISSION RESPONSE JSON =====")
                logger.info(f"  correct: {data.get('correct')}")
                logger.info(f"  reason:  {data.get('reason')}")
                logger.info(f"  url:     {data.get('url')}")
                logger.info(f"  delay:   {data.get('delay')}")
                logger.info("=" * 60)
                
                return SubmissionResponse(
                    correct=data.get("correct"),
                    message=data.get("message"),
                    url=data.get("url"),
                    reason=data.get("reason")
                )
            except Exception as parse_error:
                # If response isn't JSON, treat as message
                logger.warning(f"Failed to parse JSON: {parse_error}")
                logger.info(f"Returning raw text as message")
                return SubmissionResponse(
                    correct=None,
                    message=response.text
                )
                
        except httpx.TimeoutException:
            logger.error("Submission request timed out")
            return SubmissionResponse(
                correct=False,
                message="Request timed out"
            )
        except Exception as e:
            logger.error(f"Submission error: {e}")
            return SubmissionResponse(
                correct=False,
                message=str(e)
            )
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Global instance
action = ActionModule()
