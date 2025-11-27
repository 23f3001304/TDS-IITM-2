"""
Run test in background
"""
import subprocess
import time
import httpx
import json

print("\nTesting endpoint...")
try:
    response = httpx.post(
        "https://quiz.animy.tech/start",
        json={
            "email": "test@example.com",
            "secret": "hemang156",
            "url": "https://tds-llm-analysis.s-anand.net/demo"
        },
        timeout=3000
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
finally:
    print("\nServer stopped.")
