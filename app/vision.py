"""
Vision Module - The "Eyes"
Handles web page extraction using Selenium and BeautifulSoup
"""
import asyncio
import re
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from loguru import logger

from app.config import settings
from app.models import PageContent


class VisionModule:
    """
    Handles extraction of information from quiz URLs using Selenium.
    Implements headless Chrome browsing with JavaScript execution support.
    """
    
    def __init__(self):
        self._driver: Optional[webdriver.Chrome] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._driver_lock = asyncio.Lock() if asyncio else None
    
    def _get_driver(self) -> webdriver.Chrome:
        """Get or create Chrome WebDriver instance (reuse for speed)"""
        if self._driver is None:
            self._driver = self._create_driver()
        return self._driver
    
    def _create_driver(self) -> webdriver.Chrome:
        """Create a new Chrome WebDriver instance"""
        options = Options()
        
        if settings.chrome_headless:
            options.add_argument("--headless=new")
        
        # Essential Chrome options for stability
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Suppress logging
        options.add_argument("--log-level=3")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        try:
            # Try using Chrome directly without webdriver-manager
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(30)
            return driver
        except Exception as e:
            logger.warning(f"Direct Chrome failed: {e}, trying webdriver-manager...")
            try:
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=options)
                driver.set_page_load_timeout(30)
                return driver
            except Exception as e2:
                logger.error(f"Failed to create Chrome driver: {e2}")
                raise
    
    def _extract_page_content_sync(self, url: str) -> PageContent:
        """Synchronous page content extraction (runs in thread pool)"""
        driver = None
        try:
            driver = self._get_driver()  # Reuse driver for speed
            logger.info(f"Loading URL: {url}")
            driver.get(url)
            
            # Wait for page to load - reduced timeout
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Minimal wait for JS execution (reduced from 2s)
            import time
            time.sleep(0.5)
            
            # Get rendered HTML after JS execution
            html_content = driver.page_source
            text_content = driver.find_element(By.TAG_NAME, "body").text
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract all links
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                # Convert relative URLs to absolute
                absolute_url = urljoin(url, href)
                links.append(absolute_url)
            
            # Extract all images
            images = []
            for img_tag in soup.find_all('img', src=True):
                src = img_tag['src']
                absolute_url = urljoin(url, src)
                images.append(absolute_url)
            
            # Try to find submission endpoint
            submission_endpoint = self._find_submission_endpoint(soup, text_content, url)
            
            logger.info(f"Extracted page content: {len(text_content)} chars, {len(links)} links, {len(images)} images")
            logger.info(f"Found submission endpoint: {submission_endpoint}")
            
            return PageContent(
                url=url,
                text_content=text_content,
                links=links,
                images=images,
                submission_endpoint=submission_endpoint,
                raw_html=html_content
            )
            
        except TimeoutException:
            logger.warning(f"Timeout loading page: {url}")
            self._reset_driver()  # Reset driver on timeout
            return self._fallback_extraction(url)
        except WebDriverException as e:
            logger.warning(f"WebDriver error: {e}, using fallback")
            self._reset_driver()  # Reset driver on error
            return self._fallback_extraction(url)
        except Exception as e:
            logger.warning(f"Selenium failed: {e}, using fallback")
            return self._fallback_extraction(url)
        # Don't quit driver - reuse it for next request
    
    def _reset_driver(self):
        """Reset the driver on error"""
        if self._driver:
            try:
                self._driver.quit()
            except:
                pass
            self._driver = None
    
    def _find_submission_endpoint(self, soup: BeautifulSoup, text: str, base_url: str) -> Optional[str]:
        """Find the submission endpoint from the page content"""
        
        # FIRST: Look for <a> tags with href containing "submit" or text containing "submit"
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            link_text = a_tag.get_text().lower()
            
            # Check if href or link text contains submit-related keywords
            if 'submit' in href.lower() or 'submit' in link_text:
                absolute_url = urljoin(base_url, href)
                logger.info(f"Found submission endpoint from <a> tag: {absolute_url}")
                return absolute_url
        
        # Look for common patterns in forms
        forms = soup.find_all('form')
        for form in forms:
            action = form.get('action')
            if action:
                return urljoin(base_url, action)
        
        # Look for URLs in text - comprehensive patterns
        url_patterns = [
            # Match "POST this JSON to https://..." pattern
            r'POST\s+(?:this\s+)?(?:JSON\s+)?to\s+(https?://[^\s<>"\']+)',
            # Match "back to /submit" or "to /submit" - relative URL
            r'(?:back\s+)?to\s+(/[^\s<>"\']+submit[^\s<>"\']*)',
            # Match "submit to https://..." pattern
            r'submit\s+(?:to\s+)?(https?://[^\s<>"\']+)',
            # Match any URL containing submit
            r'(https?://[^\s<>"\']+/submit[^\s<>"\']*)',
            # Match any URL containing answer
            r'(https?://[^\s<>"\']+/answer[^\s<>"\']*)',
            # Match any URL containing api
            r'(https?://[^\s<>"\']+/api[^\s<>"\']*)',
        ]
        
        for pattern in url_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                url = match.group(1)
                # Resolve relative URLs
                absolute_url = urljoin(base_url, url)
                logger.info(f"Found submission endpoint from text pattern: {absolute_url}")
                return absolute_url
        
        # Also search in the raw HTML for href attributes
        href_pattern = r'href=["\']([^"\']*submit[^"\']*)["\']'
        match = re.search(href_pattern, str(soup), re.IGNORECASE)
        if match:
            url = match.group(1)
            absolute_url = urljoin(base_url, url)
            logger.info(f"Found submission endpoint from href: {absolute_url}")
            return absolute_url
        
        # Look in script tags for API endpoints
        scripts = soup.find_all('script')
        for script in scripts:
            script_text = script.string or ''
            for pattern in url_patterns:
                match = re.search(pattern, script_text, re.IGNORECASE)
                if match:
                    url = match.group(1)
                    absolute_url = urljoin(base_url, url)
                    logger.info(f"Found submission endpoint in script: {absolute_url}")
                    return absolute_url
        
        return None
    
    def _fallback_extraction(self, url: str) -> PageContent:
        """Fallback extraction using requests (no JS execution)"""
        import httpx
        
        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                response = client.get(url)
                html_content = response.text
                soup = BeautifulSoup(html_content, 'lxml')
                text_content = soup.get_text(separator='\n', strip=True)
                
                links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
                images = [urljoin(url, img['src']) for img in soup.find_all('img', src=True)]
                
                return PageContent(
                    url=url,
                    text_content=text_content,
                    links=links,
                    images=images,
                    submission_endpoint=self._find_submission_endpoint(soup, text_content, url),
                    raw_html=html_content
                )
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return PageContent(
                url=url,
                text_content=f"Error extracting page: {e}",
                links=[],
                images=[]
            )
    
    async def extract_page_content(self, url: str) -> PageContent:
        """
        Async wrapper for page content extraction.
        Runs Selenium in a thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._extract_page_content_sync,
            url
        )
    
    def shutdown(self):
        """Clean up resources"""
        if self._driver:
            try:
                self._driver.quit()
            except:
                pass
            self._driver = None
        self.executor.shutdown(wait=False)


# Global instance
vision = VisionModule()
