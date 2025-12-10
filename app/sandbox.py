import asyncio
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from loguru import logger

from app.config import settings
from app.models import CodeExecutionResult


class ExecutionSandbox:
    """
    Executes Python code in an isolated subprocess with timeout protection.
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.temp_dir = Path(settings.temp_dir)
        self._ensure_temp_dir()
    
    def _ensure_temp_dir(self):
        """Ensure the temp directory exists"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Sandbox temp directory: {self.temp_dir}")
    
    def _execute_code_sync(self, code: str, timeout: int = None) -> CodeExecutionResult:
        """
        Synchronously execute Python code in a subprocess.
        """
        timeout = timeout or settings.code_timeout_seconds
        
        # Create a unique file for this execution
        script_id = str(uuid.uuid4())[:8]
        script_path = self.temp_dir / f"script_{script_id}.py"
        
        try:
            # Write code to file
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            logger.info(f"Executing script: {script_path}")
            
            # Execute in subprocess with proper encoding
            result = subprocess.run(
                ['python', '-u', str(script_path)],
                capture_output=True,
                timeout=timeout,
                cwd=str(self.temp_dir),
                env={
                    **os.environ,
                    'PYTHONIOENCODING': 'utf-8',
                    'PYTHONUTF8': '1',
                    'TEMP_DIR': str(self.temp_dir),
                }
            )
            
            # Decode output with utf-8, fallback to latin-1
            try:
                stdout = result.stdout.decode('utf-8') if result.stdout else ""
            except UnicodeDecodeError:
                stdout = result.stdout.decode('latin-1', errors='replace') if result.stdout else ""
            
            try:
                stderr = result.stderr.decode('utf-8') if result.stderr else ""
            except UnicodeDecodeError:
                stderr = result.stderr.decode('latin-1', errors='replace') if result.stderr else ""
            
            logger.info(f"Execution result - return_code: {result.returncode}")
            logger.info(f"stdout: {stdout[:500] if stdout else '(empty)'}")
            if stderr:
                logger.warning(f"stderr: {stderr[:500]}")
            
            return CodeExecutionResult(
                success=result.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                return_code=result.returncode,
                timed_out=False
            )
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Code execution timed out after {timeout}s")
            return CodeExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds",
                return_code=-1,
                timed_out=True
            )
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return CodeExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                timed_out=False
            )
        finally:
            # Clean up script file
            try:
                if script_path.exists():
                    script_path.unlink()
            except:
                pass
    
    async def execute_code(self, code: str, timeout: int = None) -> CodeExecutionResult:
        """
        Async wrapper for code execution.
        Runs subprocess in thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._execute_code_sync,
            code,
            timeout
        )
    
    async def download_file(self, url: str, filename: Optional[str] = None) -> str:
        """
        Download a file to the temp directory.
        Returns the local file path.
        """
        import httpx
        
        if not filename:
            # Extract filename from URL
            from urllib.parse import urlparse, unquote
            parsed = urlparse(url)
            filename = unquote(Path(parsed.path).name) or f"download_{uuid.uuid4()[:8]}"
        
        local_path = self.temp_dir / filename
        
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded: {url} -> {local_path}")
                return str(local_path)
                
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            raise
    
    def get_temp_path(self, filename: str) -> str:
        """Get the full path for a file in the temp directory"""
        return str(self.temp_dir / filename)
    
    def cleanup(self):
        """Clean up all files in temp directory"""
        try:
            for file in self.temp_dir.iterdir():
                try:
                    if file.is_file():
                        file.unlink()
                except:
                    pass
            logger.info("Sandbox cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=False)


# Global instance
sandbox = ExecutionSandbox()
