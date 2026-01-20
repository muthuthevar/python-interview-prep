"""
FASTAPI QUICK REFERENCE - LEARNTUBE AI FOCUS
High-performance patterns for 800 req/s, sub-200ms p95
"""

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional, Dict
from datetime import datetime
import asyncio
import aiohttp
from contextlib import asynccontextmanager

# ============================================================================
# 1. BASIC FASTAPI SETUP (High Performance)
# ============================================================================

# Application lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize connections, load models
    print("Starting up...")
    # Initialize Redis, DB connections, etc.
    yield
    # Shutdown: Close connections
    print("Shutting down...")

app = FastAPI(
    title="LearnTube API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 2. PYDANTIC MODELS (Data Validation)
# ============================================================================

class QuizSubmission(BaseModel):
    user_id: int = Field(..., gt=0, description="User ID must be positive")
    quiz_id: int = Field(..., gt=0)
    answers: List[Dict[str, any]] = Field(..., min_items=1)
    timestamp: Optional[float] = Field(default=None)
    
    @validator('answers')
    def validate_answers(cls, v):
        if len(v) == 0:
            raise ValueError('Answers cannot be empty')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 123,
                "quiz_id": 456,
                "answers": [{"question_id": 1, "answer": "A"}]
            }
        }

class QuizScoreResponse(BaseModel):
    score: float = Field(..., ge=0, le=100)
    total_questions: int
    correct_answers: int
    time_taken: float
    submitted_at: datetime = Field(default_factory=datetime.now)

# ============================================================================
# 3. ASYNC ENDPOINTS (High Performance)
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "learntube-api"}

@app.get("/api/quiz/{quiz_id}")
async def get_quiz(
    quiz_id: int = Path(..., gt=0, description="Quiz ID"),
    include_answers: bool = Query(False, description="Include correct answers")
):
    """
    Get quiz by ID with optional caching
    Target: < 50ms p95
    """
    # Check cache first (Redis)
    # cached = await redis_client.get(f"quiz:{quiz_id}")
    # if cached:
    #     return json.loads(cached)
    
    # Fetch from database
    # quiz = await db.fetch_quiz(quiz_id)
    
    # Cache for 1 hour
    # await redis_client.setex(f"quiz:{quiz_id}", 3600, json.dumps(quiz))
    
    return {"quiz_id": quiz_id, "include_answers": include_answers}

@app.post("/api/quiz/submit", response_model=QuizScoreResponse)
async def submit_quiz(
    submission: QuizSubmission,
    background_tasks: BackgroundTasks
):
    """
    Submit quiz and calculate score
    Target: < 200ms p95
    Strategy: Fast path for scoring, background for analytics
    """
    # Fast path: Calculate score (synchronous, in-memory)
    score = await calculate_score_fast(submission)
    
    # Background task: Log event, update analytics (async, non-blocking)
    background_tasks.add_task(log_quiz_event, submission, score)
    
    return QuizScoreResponse(
        score=score,
        total_questions=len(submission.answers),
        correct_answers=int(score / 100 * len(submission.answers)),
        time_taken=0.0
    )

async def calculate_score_fast(submission: QuizSubmission) -> float:
    """Fast scoring algorithm - must be < 200ms"""
    # Simulate fast calculation
    await asyncio.sleep(0.01)  # Simulate DB lookup
    return 85.5  # Mock score

async def log_quiz_event(submission: QuizSubmission, score: float):
    """Background task for logging - doesn't block response"""
    # Send to SQS or write to database
    # await sqs_client.send_message(...)
    pass

# ============================================================================
# 4. DEPENDENCY INJECTION (Database, Auth, etc.)
# ============================================================================

# Database dependency
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Async engine with connection pooling
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True  # Verify connections
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db():
    """Database dependency with proper cleanup"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Authentication dependency
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return user_id
    except JWTError:
        raise credentials_exception

# Using dependencies
@app.get("/api/users/me")
async def get_current_user_info(
    current_user: int = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Protected endpoint requiring authentication"""
    # Query database
    # user = await db.get(User, current_user)
    return {"user_id": current_user}

# ============================================================================
# 5. ERROR HANDLING
# ============================================================================

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger(__name__)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom validation error handler"""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation failed",
            "details": exc.errors()
        }
    )

# ============================================================================
# 6. PERFORMANCE OPTIMIZATION
# ============================================================================

# Caching with Redis
import redis.asyncio as redis
import json

redis_client = redis.from_url("redis://localhost", decode_responses=True)

@app.get("/api/quiz/{quiz_id}/cached")
async def get_quiz_cached(quiz_id: int):
    """Cached endpoint for frequently accessed data"""
    cache_key = f"quiz:{quiz_id}"
    
    # Check cache
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Cache miss - fetch from database
    quiz_data = {"quiz_id": quiz_id, "data": "quiz content"}
    
    # Cache for 1 hour
    await redis_client.setex(
        cache_key,
        3600,
        json.dumps(quiz_data)
    )
    
    return quiz_data

# Parallel API calls
async def fetch_multiple_resources(ids: List[int]):
    """Fetch multiple resources in parallel"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_resource(session, resource_id)
            for resource_id in ids
        ]
        results = await asyncio.gather(*tasks)
    return results

async def fetch_resource(session: aiohttp.ClientSession, resource_id: int):
    """Fetch single resource"""
    async with session.get(f"https://api.example.com/resource/{resource_id}") as response:
        return await response.json()

# ============================================================================
# 7. RATE LIMITING (For 800 req/s per user)
# ============================================================================

from fastapi import Request
from fastapi.middleware.base import BaseHTTPMiddleware
import time

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        
        if client_ip not in self.clients:
            self.clients[client_ip] = []
        
        # Remove old requests
        self.clients[client_ip] = [
            t for t in self.clients[client_ip]
            if now - t < self.period
        ]
        
        if len(self.clients[client_ip]) >= self.calls:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        
        self.clients[client_ip].append(now)
        response = await call_next(request)
        return response

# Add middleware
# app.add_middleware(RateLimitMiddleware, calls=100, period=60)

# ============================================================================
# 8. MONITORING & LOGGING
# ============================================================================

from fastapi.middleware.base import BaseHTTPMiddleware
import time

class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging middleware for monitoring"""
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} - "
            f"{response.status_code} - {process_time:.3f}s"
        )
        
        # Add custom header
        response.headers["X-Process-Time"] = str(process_time)
        return response

# Add middleware
# app.add_middleware(LoggingMiddleware)

# ============================================================================
# 9. BATCH OPERATIONS (For millions of events)
# ============================================================================

class BatchQuizSubmission(BaseModel):
    submissions: List[QuizSubmission] = Field(..., max_items=100)

@app.post("/api/quiz/batch-submit")
async def batch_submit_quizzes(batch: BatchQuizSubmission):
    """
    Process multiple quiz submissions in batch
    Optimized for throughput
    """
    # Process in parallel
    tasks = [
        calculate_score_fast(submission)
        for submission in batch.submissions
    ]
    scores = await asyncio.gather(*tasks)
    
    return {
        "processed": len(batch.submissions),
        "scores": scores
    }

# ============================================================================
# 10. WEBHOOKS / EXTERNAL API INTEGRATION
# ============================================================================

@app.post("/api/webhook/quiz-completed")
async def quiz_completed_webhook(payload: dict):
    """
    Handle webhook from external system
    Process asynchronously
    """
    # Validate webhook signature
    # Process webhook data
    # Send to message queue for async processing
    return {"status": "received"}

# ============================================================================
# 11. HEALTH CHECK & METRICS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancer"""
    # Check database connection
    # Check Redis connection
    # Check external services
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Metrics endpoint for Prometheus"""
    # Return metrics in Prometheus format
    return {
        "requests_total": 1000,
        "requests_per_second": 800,
        "p95_latency_ms": 150
    }

# ============================================================================
# 12. TESTING PATTERNS
# ============================================================================

from fastapi.testclient import TestClient

client = TestClient(app)

def test_get_quiz():
    """Example test"""
    response = client.get("/api/quiz/1")
    assert response.status_code == 200
    assert "quiz_id" in response.json()

def test_submit_quiz():
    """Example test with request body"""
    response = client.post(
        "/api/quiz/submit",
        json={
            "user_id": 1,
            "quiz_id": 1,
            "answers": [{"question_id": 1, "answer": "A"}]
        }
    )
    assert response.status_code == 200
    assert "score" in response.json()

# ============================================================================
# KEY PERFORMANCE TIPS FOR LEARNTUBE
# ============================================================================

"""
1. USE ASYNC/AWAIT for all I/O operations
   - Database queries
   - External API calls
   - Redis operations

2. CACHE AGGRESSIVELY
   - Quiz data (Redis, 1 hour TTL)
   - User sessions (Redis, 30 min TTL)
   - Frequently accessed data

3. USE BACKGROUND TASKS
   - Analytics logging
   - Email notifications
   - Heavy computations

4. CONNECTION POOLING
   - Database: pool_size=20, max_overflow=10
   - Redis: Connection pool
   - HTTP clients: Session reuse

5. BATCH OPERATIONS
   - Process multiple items in parallel
   - Use asyncio.gather() for parallel execution

6. MONITOR EVERYTHING
   - Response times (p50, p95, p99)
   - Error rates
   - Request throughput

7. OPTIMIZE DATABASE QUERIES
   - Use indexes
   - Avoid N+1 queries
   - Use select_related/prefetch_related

8. RATE LIMITING
   - Per user/IP
   - Use Redis for distributed rate limiting

9. ERROR HANDLING
   - Graceful degradation
   - Retry logic with exponential backoff
   - Circuit breakers for external services

10. SCALING
    - Horizontal scaling (multiple instances)
    - Load balancing (NGINX)
    - Stateless services
    - Shared cache (Redis)
"""

if __name__ == "__main__":
    import uvicorn
    # Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
    uvicorn.run(app, host="0.0.0.0", port=8000)
