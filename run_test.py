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
        "http://127.0.0.1:8000/start",
        json={
            "email": "23f301035@ds.study.iitm.ac.in",
            "secret": "hemang156",
            "url": "https://tds-llm-analysis.s-anand.net/project2"
        },
        timeout=3000
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
finally:
    print("\nServer stopped.")
