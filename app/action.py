"""
Action Module - The "Mouth"
Handles submission of answers to the quiz server
"""
import httpx
from typing import Any, Optional
from loguru import logger

from app.models import SubmissionPayload, SubmissionResponse


class ActionModule:
    """
    Handles communication with the quiz server for answer submission.
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30, follow_redirects=True)
    
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
            
            try:
                data = response.json()
                logger.info(f"Response data: {data}")
                
                return SubmissionResponse(
                    correct=data.get("correct"),
                    message=data.get("message"),
                    url=data.get("url"),
                    reason=data.get("reason")
                )
            except:
                # If response isn't JSON, treat as message
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
