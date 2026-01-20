# 🎯 LEARNTUBE AI - TARGETED INTERVIEW PREP

## 📋 JOB REQUIREMENTS ANALYSIS

### Must-Haves (Will be tested):
1. ✅ **3+ yrs Python back-end experience (FastAPI)** - CRITICAL
2. ✅ **Docker & container orchestration** - Important
3. ✅ **GitLab CI/CD, AWS/GCP** - May be asked
4. ✅ **SQL/NoSQL (Postgres, MongoDB)** - CRITICAL
5. ✅ **System design fundamentals** - CRITICAL
6. ✅ **Built systems from scratch** - System design questions

### Nice-to-Haves (Bonus):
- Kubernetes at scale
- AI/ML inference services (LLMs, vector DBs)
- Observability (Prometheus, Grafana)
- Go/Rust for high-perf services

### Key Metrics They Care About:
- **800 req/s** (scaling to 2400 req/s)
- **Sub-200ms p95 latency**
- **99.9% uptime**
- **Millions of events daily**
- **Multiple prod deploys per week**

---

## 🎯 EQUIP.CO FIRST ROUND - WHAT TO EXPECT

### Format (Based on Research):
1. **Multiple Choice Questions** (Python, FastAPI, System Design)
2. **Coding Problems** (2-3 problems, 60-90 minutes)
3. **Possibly System Design** (for senior role)

### Topics They'll Test:

#### 1. Python & FastAPI (HIGH PRIORITY)
- FastAPI fundamentals
- Async/await patterns
- Pydantic models
- Dependency injection
- Error handling
- Performance optimization

#### 2. Data Structures & Algorithms (HIGH PRIORITY)
- Arrays, strings, hash maps
- Trees, graphs
- Dynamic programming
- Two pointers, sliding window
- Binary search

#### 3. Database & SQL (MEDIUM-HIGH PRIORITY)
- PostgreSQL queries
- MongoDB operations
- Redis usage
- Query optimization
- Indexing strategies

#### 4. System Design Basics (MEDIUM PRIORITY)
- Microservices architecture
- Scalability patterns
- Caching strategies
- Database design
- API design

---

## 📚 TARGETED STUDY PLAN FOR EQUIP.CO ROUND

### 🔴 TIER 1: CRITICAL (Focus 70% of time here)

#### 1. **EQUIP_CO_PRACTICE.py** ⭐⭐⭐⭐⭐
   - **Why**: Exact type of coding problems Equip.co asks
   - **Focus**: Arrays, strings, hash maps, two pointers
   - **Time**: 3 hours

#### 2. **CRITICAL_DSA_MUST_KNOW.py** ⭐⭐⭐⭐⭐
   - **Why**: Essential DSA patterns
   - **Focus**: All patterns, especially DP and binary search
   - **Time**: 3 hours

#### 3. **FastAPI Deep Dive** (from 07_api_web_development.md) ⭐⭐⭐⭐⭐
   - **Why**: Job requires FastAPI, will definitely be tested
   - **Focus**: 
     - Async endpoints
     - Pydantic models
     - Dependency injection
     - Error handling
     - Performance optimization
   - **Time**: 2 hours

#### 4. **SQL_PRACTICE.sql** ⭐⭐⭐⭐
   - **Why**: Postgres is must-have
   - **Focus**: PostgreSQL queries, optimization, window functions
   - **Time**: 1.5 hours

---

### 🟠 TIER 2: IMPORTANT (Focus 20% of time)

#### 5. **System Design Basics** (from 05_system_design.md) ⭐⭐⭐⭐
   - **Why**: Job requires building systems from scratch
   - **Focus**:
     - Microservices architecture
     - Scalability (800 req/s → 2400 req/s)
     - Caching strategies (Redis)
     - Database design (Postgres, MongoDB)
     - Message queues (SQS)
   - **Time**: 1.5 hours

#### 6. **Python Fundamentals** (from 01_python_fundamentals.md) ⭐⭐⭐
   - **Why**: Foundation for everything
   - **Focus**: Decorators, generators, async/await, GIL
   - **Time**: 1 hour

#### 7. **DSA_PATTERNS_CHEATSHEET.md** ⭐⭐⭐⭐⭐
   - **Why**: Quick pattern recognition
   - **Time**: 30 min (review)

---

### 🟡 TIER 3: BONUS (Focus 10% of time)

#### 8. **Concurrency & Async** (from 08_concurrency_async.md) ⭐⭐⭐
   - **Why**: FastAPI uses async, high-performance requirements
   - **Focus**: Async/await, when to use threading vs async
   - **Time**: 1 hour

#### 9. **Database Deep Dive** (from 06_database_sql.md) ⭐⭐
   - **Why**: Postgres + MongoDB required
   - **Time**: Reference only

---

## 🚀 FASTAPI - CRITICAL TOPICS FOR LEARNTUBE

### 1. High-Performance FastAPI Patterns

```python
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio
from contextlib import asynccontextmanager

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown

app = FastAPI(lifespan=lifespan)

# CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models with validation
class QuizSubmission(BaseModel):
    user_id: int = Field(..., gt=0)
    quiz_id: int = Field(..., gt=0)
    answers: List[dict] = Field(..., min_items=1)
    timestamp: Optional[float] = None

class QuizScoreResponse(BaseModel):
    score: float = Field(..., ge=0, le=100)
    total_questions: int
    correct_answers: int
    time_taken: float

# Async endpoint for high performance
@app.post("/api/quiz/submit", response_model=QuizScoreResponse)
async def submit_quiz(
    submission: QuizSubmission,
    background_tasks: BackgroundTasks
):
    """
    Handle quiz submission with sub-200ms p95 requirement
    """
    # Process scoring (fast path)
    score = await calculate_score(submission)
    
    # Background task for analytics
    background_tasks.add_task(log_quiz_event, submission, score)
    
    return QuizScoreResponse(
        score=score,
        total_questions=len(submission.answers),
        correct_answers=int(score / 100 * len(submission.answers)),
        time_taken=0.0
    )

# Dependency injection for database
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

engine = create_async_engine("postgresql+asyncpg://...")
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Using dependency
@app.get("/api/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

### 2. Performance Optimization

```python
# Async database operations
async def calculate_score(quiz_submission: QuizSubmission) -> float:
    # Use async for I/O operations
    async with aiohttp.ClientSession() as session:
        # Parallel API calls if needed
        tasks = [fetch_question_data(q_id) for q_id in question_ids]
        results = await asyncio.gather(*tasks)
    
    # Fast in-memory calculation
    return compute_score(results)

# Caching with Redis
import redis.asyncio as redis

redis_client = redis.from_url("redis://localhost")

@app.get("/api/quiz/{quiz_id}")
async def get_quiz(quiz_id: int):
    # Check cache first
    cached = await redis_client.get(f"quiz:{quiz_id}")
    if cached:
        return json.loads(cached)
    
    # Fetch from database
    quiz = await fetch_quiz_from_db(quiz_id)
    
    # Cache for 1 hour
    await redis_client.setex(
        f"quiz:{quiz_id}",
        3600,
        json.dumps(quiz)
    )
    return quiz
```

### 3. Error Handling & Monitoring

```python
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger(__name__)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation failed", "details": exc.errors()}
    )

# Middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response
```

---

## 🗄️ DATABASE - LEARNTUBE SPECIFIC

### PostgreSQL (Primary Database)

```sql
-- Optimized query for quiz scoring (millions of events)
CREATE INDEX idx_quiz_submissions_user_quiz ON quiz_submissions(user_id, quiz_id, submitted_at DESC);

-- Query with window function for analytics
SELECT 
    user_id,
    quiz_id,
    score,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY submitted_at DESC) as attempt_number,
    AVG(score) OVER (PARTITION BY user_id) as avg_score
FROM quiz_submissions
WHERE user_id = $1
ORDER BY submitted_at DESC
LIMIT 10;

-- Partitioning for scale (millions of events)
CREATE TABLE quiz_submissions (
    id BIGSERIAL,
    user_id INT NOT NULL,
    quiz_id INT NOT NULL,
    score FLOAT NOT NULL,
    submitted_at TIMESTAMP NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (submitted_at);

-- Create monthly partitions
CREATE TABLE quiz_submissions_2024_01 PARTITION OF quiz_submissions
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### MongoDB (For Flexible Data)

```python
from pymongo import MongoClient
from pymongo.collection import Collection

# Connection pooling for performance
client = MongoClient(
    "mongodb://localhost:27017/",
    maxPoolSize=50,
    minPoolSize=10
)

# Efficient queries
async def get_user_learning_path(user_id: int):
    collection = client.learntube.user_paths
    
    # Indexed query
    path = collection.find_one(
        {"user_id": user_id},
        {"_id": 0, "path": 1, "progress": 1}
    )
    
    # Aggregation for analytics
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {
            "_id": "$category",
            "total_time": {"$sum": "$time_spent"},
            "avg_score": {"$avg": "$score"}
        }}
    ]
    results = collection.aggregate(pipeline)
    return list(results)
```

### Redis (Caching & Real-time)

```python
import redis.asyncio as redis

redis_client = redis.from_url("redis://localhost", decode_responses=True)

# Cache quiz data
async def get_quiz_cached(quiz_id: int):
    key = f"quiz:{quiz_id}"
    cached = await redis_client.get(key)
    if cached:
        return json.loads(cached)
    
    quiz = await fetch_quiz(quiz_id)
    await redis_client.setex(key, 3600, json.dumps(quiz))
    return quiz

# Real-time leaderboard
async def update_leaderboard(user_id: int, score: float):
    await redis_client.zadd("leaderboard", {str(user_id): score})
    # Get top 10
    top_10 = await redis_client.zrevrange("leaderboard", 0, 9, withscores=True)
    return top_10
```

---

## 🏗️ SYSTEM DESIGN - LEARNTUBE SCALE

### Architecture for 800 req/s → 2400 req/s

```
┌─────────────┐
│   NGINX     │  Load Balancer
│  (Reverse   │
│   Proxy)    │
└──────┬──────┘
       │
       ├─────────────────┬─────────────────┐
       │                 │                 │
┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
│  FastAPI    │  │  FastAPI    │  │  FastAPI    │
│  Service 1  │  │  Service 2  │  │  Service 3  │
│  (K8s Pod)  │  │  (K8s Pod)  │  │  (K8s Pod)  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼────┐    ┌─────▼─────┐    ┌────▼────┐
   │ Redis   │    │ Postgres  │    │  SQS    │
   │ (Cache) │    │ (Primary) │    │ (Queue) │
   └─────────┘    └───────────┘    └─────────┘
                         │
                   ┌─────▼─────┐
                   │ MongoDB   │
                   │ (Flexible)│
                   └───────────┘
```

### Key Design Decisions:

1. **Microservices**: Separate services for quiz scoring, AI tutor, user management
2. **Caching**: Redis for hot data (quiz data, user sessions)
3. **Database**: Postgres for structured data, MongoDB for flexible schemas
4. **Message Queue**: SQS for async processing (analytics, notifications)
5. **Load Balancing**: NGINX for request distribution
6. **Containerization**: Docker + Kubernetes for scaling

### Scaling Strategy:

```python
# Horizontal scaling with Kubernetes
# Each FastAPI service can scale independently

# Example: Quiz scoring service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quiz-scoring-service
spec:
  replicas: 3  # Scale based on load
  template:
    spec:
      containers:
      - name: fastapi
        image: learntube/quiz-scoring:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

---

## 🎯 EQUIP.CO ASSESSMENT - EXPECTED QUESTIONS

### Coding Problems (Most Likely):

1. **Quiz Scoring Algorithm**
   - Given quiz answers and correct answers, calculate score
   - Handle edge cases (partial credit, time bonus)
   - Optimize for performance

2. **User Progress Tracking**
   - Design data structure to track user progress
   - Efficient queries for progress retrieval
   - Handle concurrent updates

3. **API Rate Limiting**
   - Implement rate limiter for API endpoints
   - Handle 800 req/s per user
   - Use Redis for distributed rate limiting

4. **Event Processing**
   - Process millions of quiz events
   - Batch processing
   - Error handling and retries

### FastAPI Questions:

1. How do you handle async operations in FastAPI?
2. How do you optimize FastAPI for high throughput?
3. How do you implement caching in FastAPI?
4. How do you handle database connections in FastAPI?
5. How do you structure a FastAPI microservice?

### System Design Questions:

1. Design a quiz scoring system that handles 800 req/s
2. Design a system to track user learning progress
3. How would you scale from 800 to 2400 req/s?
4. Design a caching strategy for quiz data
5. How would you ensure 99.9% uptime?

---

## ✅ FINAL CHECKLIST FOR EQUIP.CO ROUND

### Before Assessment:
- [ ] Master top 15 DSA problems
- [ ] Understand FastAPI async patterns
- [ ] Know PostgreSQL optimization
- [ ] Understand microservices architecture
- [ ] Practice system design basics
- [ ] Review Redis caching patterns
- [ ] Understand SQS message queues

### During Assessment:
- [ ] Read problems carefully
- [ ] Identify patterns quickly
- [ ] Write clean, optimized code
- [ ] Handle edge cases
- [ ] Explain your approach
- [ ] Test with examples

### Key Points to Remember:
- **Performance matters**: Sub-200ms p95 latency
- **Scale matters**: 800 → 2400 req/s
- **Reliability matters**: 99.9% uptime
- **FastAPI expertise**: This is critical
- **System design**: Building from scratch

---

## 🚀 QUICK WINS - LAST MINUTE PREP

### FastAPI Cheat Sheet:
```python
# Async endpoint
@app.get("/api/endpoint")
async def endpoint():
    return {"data": "value"}

# Dependency injection
def get_db(): yield db

@app.get("/users")
async def get_users(db = Depends(get_db)):
    return db.query(User).all()

# Background tasks
@app.post("/submit")
async def submit(background_tasks: BackgroundTasks):
    background_tasks.add_task(process_data)
    return {"status": "accepted"}

# Error handling
raise HTTPException(status_code=404, detail="Not found")
```

### Performance Tips:
- Use async/await for I/O operations
- Cache frequently accessed data (Redis)
- Use connection pooling for databases
- Batch database operations
- Use background tasks for heavy processing

### System Design Tips:
- Start with requirements (800 req/s, sub-200ms)
- Identify bottlenecks
- Use caching (Redis)
- Scale horizontally (multiple instances)
- Use message queues for async processing
- Monitor everything (Prometheus/Grafana)

**Good luck! Focus on FastAPI, DSA, and system design basics! 🚀**
