"""
Quiz Solver - Main solving logic
"""
import hashlib
import re
from typing import Any, Optional
from urllib.parse import urlparse

from loguru import logger

from app.config import settings
from app.models import PageContent, QuizResult
from app.vision import vision
from app.action import action

from app.agent.models import QuizContext, QuizDependencies, QuizAnswer
from app.agent.prompts import quiz_agent, guidance_agent


class QuizSolver:
    """Main quiz solver using Pydantic AI agent"""

    async def solve_quiz(self, context: QuizContext) -> list[QuizResult]:
        """Main entry point - solves a quiz chain starting from the initial URL."""
        logger.info(f"Starting quiz solving for {context.email}")
        logger.info(f"Initial URL: {context.current_url}")

        while True:
            try:
                result = await self._solve_single_question(context)
                context.results.append(result)

                if result.correct:
                    if result.next_url:
                        logger.info(f"Correct! Moving to next question: {result.next_url}")
                        context.current_url = result.next_url
                        context.attempt_number = 0
                        context.last_failure_reason = ""
                        context.last_wrong_answer = ""
                    else:
                        logger.info("Quiz completed successfully!")
                        break
                else:
                    reason = (result.message or "").lower()
                    is_delay_timeout = "delay" in reason and "180" in reason
                    
                    context.last_failure_reason = result.message or "Wrong answer"
                    context.last_wrong_answer = str(result.answer) if result.answer else ""
                    
                    if result.next_url:
                        if is_delay_timeout:
                            logger.info(f"Delay timeout, moving to next: {result.next_url}")
                            context.current_url = result.next_url
                            context.attempt_number = 0
                            context.last_failure_reason = ""
                            context.last_wrong_answer = ""
                        else:
                            context.attempt_number += 1
                            if context.attempt_number >= settings.max_retries_per_question:
                                logger.warning(f"Max retries reached, moving to next: {result.next_url}")
                                context.current_url = result.next_url
                                context.attempt_number = 0
                                context.last_failure_reason = ""
                                context.last_wrong_answer = ""
                            else:
                                logger.info(f"Wrong answer '{context.last_wrong_answer}', reason: '{context.last_failure_reason}', retrying (attempt {context.attempt_number + 1}/{settings.max_retries_per_question})")
                                context.results.pop()
                                continue
                    else:
                        if is_delay_timeout:
                            logger.info("Delay timeout and no next URL, quiz ended")
                            break
                        context.attempt_number += 1
                        if context.attempt_number >= settings.max_retries_per_question:
                            logger.warning(f"Max retries reached, no next URL, quiz ended")
                            break
                        logger.info(f"Wrong answer '{context.last_wrong_answer}', reason: '{context.last_failure_reason}', retrying (attempt {context.attempt_number + 1}/{settings.max_retries_per_question})")
                        context.results.pop()
                        continue

            except Exception as e:
                logger.error(f"Error solving question: {e}", exc_info=True)
                try:
                    emergency_result = await self._emergency_submit(context)
                    context.results.append(emergency_result)
                    if emergency_result.next_url:
                        logger.info(f"Emergency submit got next URL: {emergency_result.next_url}")
                        context.current_url = emergency_result.next_url
                        context.attempt_number = 0
                        continue
                except Exception as e2:
                    logger.error(f"Emergency submit also failed: {e2}")
                
                context.results.append(QuizResult(
                    url=context.current_url,
                    answer=None,
                    correct=False,
                    message=str(e)
                ))
                context.attempt_number += 1
                if context.attempt_number >= settings.max_retries_per_question:
                    logger.warning(f"Max retries reached on error, stopping")
                    break

        return context.results

    async def _emergency_submit(self, context: QuizContext) -> QuizResult:
        """Emergency submission to try to get a next_url when agent fails."""
        url = context.current_url
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        response = await action.submit_answer(
            endpoint=f"{base_url}/submit",
            email=context.email,
            secret=context.secret,
            url=url,
            answer="error_fallback"
        )
        
        return QuizResult(
            url=url,
            answer="error_fallback",
            correct=response.correct,
            message=response.message or response.reason,
            next_url=response.url
        )

    async def _solve_single_question(self, context: QuizContext) -> QuizResult:
        url = context.current_url
        logger.info(f"Solving question: {url}")
        
        page = None
        try:
            page = await vision.extract_page_content(url)
            if not page or not page.text_content:
                page = await vision.extract_page_content(url)
            
            if not page:
                raise RuntimeError(f"Failed to extract page content from {url}")

            logger.info(f"Page text: {page.text_content[:300]}...")
            logger.info(f"Links: {page.links}")

            deps = QuizDependencies(
                email=context.email,
                secret=context.secret,
                current_url=url,
                page_content=page
            )

            guidance = ""
            try:
                logger.info("Getting solution guidance...")
                guidance = await self._get_solution_guidance(page)
                logger.info(f"Guidance received ({len(guidance)} chars): {guidance}")
            except Exception as e:
                logger.warning(f"Guidance failed: {type(e).__name__}: {e}, continuing without it")

            retry_context = ""
            if context.attempt_number > 0 and context.last_failure_reason:
                retry_context = f"""
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️ RETRY ATTEMPT {context.attempt_number + 1}/{settings.max_retries_per_question} - YOUR PREVIOUS ANSWER WAS WRONG ⚠️  ║
╚══════════════════════════════════════════════════════════════════╝

YOUR REJECTED ANSWER:
{context.last_wrong_answer}

SERVER FEEDBACK:
{context.last_failure_reason}

CRITICAL INSTRUCTIONS FOR THIS RETRY:
1. DO NOT submit the same answer again - it was already rejected
2. Re-read any schema/config files and match their EXACT format
3. Check field names carefully (e.g., "name" vs "tool", "args" structure)
4. If server says format is wrong, examine the schema and use EXACTLY what it shows
5. Your new answer MUST be different from the rejected one above
"""
                logger.info(f"Adding retry context to prompt: previous answer='{context.last_wrong_answer}', reason='{context.last_failure_reason}'")
            
            prompt = self._build_prompt(url, page, deps, guidance, retry_context)
            logger.info(f"Built prompt, length: {len(prompt)}")

            try:
                logger.info("Running quiz agent...")
                import traceback
                try:
                    result = await quiz_agent.run(prompt, deps=deps)
                except Exception as inner_e:
                    logger.error(f"INNER EXCEPTION: {type(inner_e).__name__}: {inner_e}")
                    logger.error(f"INNER TRACEBACK:\n{traceback.format_exc()}")
                    raise
                logger.info(f"Quiz agent returned: {type(result)}")
                
                if result is None or result.output is None:
                    logger.warning("Agent returned None, using fallback")
                    return await self._fallback_solve(context, page, deps)
                
                agent_answer = result.output
                logger.info(f"Agent answer type: {type(agent_answer)}, answer: {agent_answer.answer}")
                
                if isinstance(agent_answer.answer, str) and agent_answer.answer.lower() in ['error', "'error'", '"error"']:
                    logger.warning(f"Agent returned error answer: {agent_answer.answer}, using fallback")
                    return await self._fallback_solve(context, page, deps)

                final_answer = self._postprocess_answer(agent_answer.answer, page.text_content)

                logger.info(f"Agent result: answer={final_answer}, submission_url={agent_answer.submission_url}")

                submission_endpoint = self._resolve_submission_url(agent_answer, page, deps)

                response = await action.submit_answer(
                    endpoint=submission_endpoint,
                    email=context.email,
                    secret=context.secret,
                    url=url,
                    answer=final_answer
                )

                return QuizResult(
                    url=url,
                    answer=final_answer,
                    correct=response.correct,
                    message=response.message or response.reason,
                    next_url=response.url
                )

            except KeyError as ke:
                logger.error(f"KeyError in agent run: {ke}")
                logger.exception("Full KeyError traceback:")
                return await self._fallback_solve(context, page, deps)

            except Exception as e:
                logger.error(f"Agent failed: {type(e).__name__}: {e}", exc_info=True)
                return await self._fallback_solve(context, page, deps)
        
        except Exception as outer_e:
            logger.error(f"Question solving failed completely: {type(outer_e).__name__}: {outer_e}", exc_info=True)
            try:
                parsed = urlparse(url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                response = await action.submit_answer(
                    endpoint=f"{base_url}/submit",
                    email=context.email,
                    secret=context.secret,
                    url=url,
                    answer="fallback_error"
                )
                return QuizResult(
                    url=url,
                    answer="fallback_error",
                    correct=response.correct,
                    message=response.message or response.reason,
                    next_url=response.url
                )
            except:
                raise outer_e

    def _postprocess_answer(self, answer: Any, page_text: str) -> Any:
        """Post-process the answer to fix common issues."""
        if not isinstance(answer, str):
            return answer
        
        if answer.startswith('uv http get "') or answer.startswith("uv http get '"):
            answer = re.sub(r'^(uv http get )["\']([^"\']+)["\'](.*)$', r'\1\2\3', answer)
            logger.info(f"Postprocess: removed quotes from uv command -> {answer}")
        
        return answer

    async def _get_solution_guidance(self, page: PageContent) -> str:
        """Get solution guidance from the guidance agent before solving."""
        try:
            guidance_prompt = f"""Analyze this quiz question and provide a brief solution strategy:

QUESTION:
{page.text_content}

AVAILABLE FILES/LINKS:
{page.links if page.links else "(none)"}

Provide 3-5 bullet points on:
- Problem type and approach
- Tools needed (if any)
- Key requirements/format rules
- Potential gotchas
"""
            logger.info("Calling guidance agent...")
            result = await guidance_agent.run(guidance_prompt)
            guidance_text = result.output
            logger.info(f"Guidance agent returned: {guidance_text}")
            return guidance_text
        except Exception as e:
            logger.warning(f"Guidance agent failed: {type(e).__name__}: {e}")
            return ""

    def _build_prompt(self, url: str, page: PageContent, deps: QuizDependencies, guidance: str = "", retry_context: str = "") -> str:
        links_formatted = (
            "\n".join([f"  - {link}" for link in page.links])
            if page.links else "  (none)"
        )

        guidance_section = ""
        if guidance:
            guidance_section = f"""
SOLUTION STRATEGY (follow this guidance):
{guidance}

"""

        retry_section = ""
        if retry_context:
            retry_section = f"""
{retry_context}
"""

        return f"""Answer this question. DO NOT use tools unless the question requires downloading/analyzing a file.

Email: {deps.email}
{retry_section}{guidance_section}
QUESTION:
{page.text_content}

AVAILABLE FILES (only use if question asks to download/analyze):
{links_formatted}

Return your answer directly. For command questions, just return the command string with {deps.email} replacing <your email>.
Set submission_url to: {deps.base_url}/submit
"""

    def _resolve_submission_url(
        self,
        agent_answer: QuizAnswer,
        page: PageContent,
        deps: QuizDependencies
    ) -> str:
        """Resolve the submission URL from agent answer or fallbacks."""
        submission_endpoint = None
        
        if page.submission_endpoint and 'submit' in page.submission_endpoint.lower():
            submission_endpoint = page.submission_endpoint
            logger.info(f"Using vision submission endpoint: {submission_endpoint}")
        
        if not submission_endpoint and page.links:
            for link in page.links:
                if 'submit' in link.lower():
                    submission_endpoint = link
                    logger.info(f"Using submit link from page: {submission_endpoint}")
                    break
        
        if not submission_endpoint:
            submission_endpoint = f"{deps.base_url}/submit"
            logger.info(f"Using /submit endpoint: {submission_endpoint}")
        
        return submission_endpoint

    async def _fallback_solve(
        self,
        context: QuizContext,
        page: PageContent,
        deps: QuizDependencies
    ) -> QuizResult:
        """Fallback solver when AI agent fails."""
        logger.info("Using fallback solver")

        url = context.current_url
        text_lower = page.text_content.lower()
        submission_endpoint = page.submission_endpoint or f"{deps.base_url}/submit"
        answer = None

        if "anything you want" in text_lower or "any answer" in text_lower:
            answer = "test_answer"
        elif "scrape" in text_lower:
            answer = await self._handle_scrape_fallback(page, url)
        elif "email" in text_lower and "code" in text_lower:
            sha1_hash = hashlib.sha1(context.email.encode()).hexdigest()
            answer = str(int(sha1_hash[:4], 16))
        elif "hash" in text_lower:
            match = re.search(r'hash\s+of\s+["\']?(\w+)["\']?', text_lower)
            if match:
                answer = hashlib.sha256(match.group(1).encode()).hexdigest()

        if not answer:
            answer = "unknown"

        logger.info(f"Fallback answer: {answer}")

        response = await action.submit_answer(
            endpoint=submission_endpoint,
            email=context.email,
            secret=context.secret,
            url=url,
            answer=answer
        )

        return QuizResult(
            url=url,
            answer=answer,
            correct=response.correct,
            message=response.message or response.reason,
            next_url=response.url
        )

    async def _handle_scrape_fallback(self, page: PageContent, current_url: str) -> Optional[str]:
        """Handle scraping in fallback mode."""
        scrape_urls = [u for u in page.links if 'submit' not in u.lower() and u != current_url]

        if not scrape_urls:
            return None

        for scrape_url in scrape_urls[:3]:
            try:
                scraped_page = await vision.extract_page_content(scrape_url)
                scraped_text = scraped_page.text_content
                logger.info(f"Fallback scraped: {scraped_text[:200]}")

                patterns = [
                    r'secret\s+code\s+is\s+(\d+)',
                    r'code\s*[:\-=]\s*(\d+)',
                    r'answer\s*[:\-=]\s*(\d+)',
                    r'result\s*[:\-=]\s*(\d+)',
                ]

                for pattern in patterns:
                    match = re.search(pattern, scraped_text, re.IGNORECASE)
                    if match:
                        answer = match.group(1)
                        logger.info(f"Extracted code: {answer}")
                        return answer

                numbers = re.findall(r'\b(\d{4,6})\b', scraped_text)
                if numbers:
                    answer = numbers[0]
                    logger.info(f"Found number as fallback: {answer}")
                    return answer

            except Exception as e:
                logger.error(f"Fallback scrape failed for {scrape_url}: {e}")
                continue

        return None
