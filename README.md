# Autonomous Analyst

AI-powered quiz solving API using the ReAct pattern. Built with FastAPI, Google's Gemini AI, and Selenium for browser automation.

## Features

- **AI-Powered Problem Solving**: Uses Google's Gemini AI to analyze and solve quiz questions
- **Browser Automation**: Selenium-based web scraping and interaction
- **Sandboxed Code Execution**: Safe Python code execution environment
- **Asynchronous Processing**: Background task processing for non-blocking API responses
- **RESTful API**: Clean FastAPI-based endpoints
- **Docker Ready**: Containerized deployment with Docker and Docker Compose
- **Heroku Compatible**: Easy deployment to Heroku platform

## Requirements

- Python 3.11+
- Google API Key (for Gemini AI) - [Get one from Google AI Studio](https://aistudio.google.com/app/apikey)
- Chrome/Chromium (for Selenium)

## Installation

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/23f3001304/TDS-IITM-2.git
   cd TDS-IITM-2
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   SECRET_KEY=your_secret_key
   LOG_LEVEL=INFO
   ```

5. Run the server:
   ```bash
   python main.py
   ```

### Docker Setup

1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

   Or using Docker directly:
   ```bash
   docker build -t autonomous-analyst .
   docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key -e SECRET_KEY=your_secret autonomous-analyst
   ```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `SECRET_KEY` | API authentication secret | Required (set a secure value) |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `CHROME_HEADLESS` | Run Chrome in headless mode | `true` |
| `CODE_TIMEOUT_SECONDS` | Code execution timeout | `60` |
| `QUIZ_TIMEOUT_SECONDS` | Quiz solving timeout | `180` |
| `MAX_RETRIES_PER_QUESTION` | Max retries per question | `2` |

## API Endpoints

### POST /start
Start a quiz solving session (asynchronous).

**Request:**
```json
{
  "email": "user@example.com",
  "secret": "your_secret_key",
  "url": "https://quiz-url.com/question"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Secret verified. Quiz solving started.",
  "task_id": "uuid-string"
}
```

### GET /status/{task_id}
Get the status of a quiz solving task.

**Response:**
```json
{
  "status": "completed",
  "email": "user@example.com",
  "url": "https://quiz-url.com/question",
  "started_at": 1234567890,
  "results": [...],
  "error": null
}
```

### POST /solve-sync
Synchronous quiz solving (blocks until complete).

**Request:** Same as `/start`

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Project Structure

```
├── main.py              # FastAPI application entry point
├── app/
│   ├── __init__.py
│   ├── action.py        # HTTP actions and submissions
│   ├── agent.py         # Quiz solving agent (ReAct pattern)
│   ├── config.py        # Application settings
│   ├── models.py        # Pydantic models
│   ├── sandbox.py       # Code execution sandbox
│   └── vision.py        # Vision/image processing
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
├── heroku.yml           # Heroku deployment config
└── run_test.py          # Test script
```

## Deployment

### Heroku

1. Create a Heroku app with container stack:
   ```bash
   heroku create your-app-name
   heroku stack:set container
   ```

2. Set environment variables:
   ```bash
   heroku config:set GOOGLE_API_KEY=your_key SECRET_KEY=your_secret
   ```

3. Deploy:
   ```bash
   git push heroku main
   ```

## License

MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2025 Hemang Choudhary
