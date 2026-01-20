# API & Web Development - Interview Questions

## Web Frameworks

### 1. Flask
- **Lightweight Framework**
  - Minimal core, extensible
  - Jinja2 templating
  - Werkzeug WSGI toolkit

- **Key Concepts**
  - Routes and view functions
  - Request/Response objects
  - Templates
  - Blueprints
  - Application factory pattern

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify({'users': []})

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    return jsonify({'id': user_id, 'name': 'Alice'})

@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json()
    # Process data
    return jsonify({'id': 1, **data}), 201

if __name__ == '__main__':
    app.run(debug=True)
```

### 2. FastAPI
- **Modern Framework**
  - Built on Starlette and Pydantic
  - Automatic API documentation
  - Type hints and validation
  - Async support

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

class UserCreate(BaseModel):
    name: str
    email: str

@app.get("/api/users", response_model=List[User])
async def get_users():
    return []

@app.get("/api/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    return users[user_id]

@app.post("/api/users", response_model=User, status_code=201)
async def create_user(user: UserCreate):
    new_user = User(id=len(users)+1, **user.dict())
    users.append(new_user)
    return new_user
```

### 3. Django
- **Full-Stack Framework**
  - ORM
  - Admin interface
  - Authentication
  - URL routing
  - Template engine

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def user_list(request):
    if request.method == 'GET':
        return JsonResponse({'users': []})
    elif request.method == 'POST':
        data = json.loads(request.body)
        # Create user
        return JsonResponse({'id': 1, **data}, status=201)

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/users/', views.user_list),
]
```

## RESTful API Design

### 1. REST Principles
- **Stateless**: Each request contains all information
- **Resource-Based**: URLs represent resources
- **HTTP Methods**: GET, POST, PUT, PATCH, DELETE
- **Status Codes**: Proper HTTP status codes

### 2. HTTP Methods
- **GET**: Retrieve resource (idempotent, safe)
- **POST**: Create resource (not idempotent)
- **PUT**: Update/replace resource (idempotent)
- **PATCH**: Partial update (idempotent)
- **DELETE**: Delete resource (idempotent)

### 3. Status Codes
- **2xx Success**
  - 200 OK
  - 201 Created
  - 204 No Content

- **4xx Client Error**
  - 400 Bad Request
  - 401 Unauthorized
  - 403 Forbidden
  - 404 Not Found
  - 409 Conflict
  - 422 Unprocessable Entity

- **5xx Server Error**
  - 500 Internal Server Error
  - 502 Bad Gateway
  - 503 Service Unavailable

### 4. API Design Best Practices
- **URL Structure**
  - Use nouns, not verbs
  - Use plural nouns
  - Hierarchical structure
  - Example: `/api/users/123/posts`

- **Versioning**
  - URL versioning: `/api/v1/users`
  - Header versioning: `Accept: application/vnd.api+json;version=1`

- **Pagination**
  - Limit and offset
  - Cursor-based pagination
  - Include metadata (total, page, per_page)

- **Filtering and Sorting**
  - Query parameters: `?status=active&sort=created_at`
  - Consistent parameter names

## Authentication & Authorization

### 1. Authentication Methods

#### API Keys
```python
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function
```

#### JWT (JSON Web Tokens)
```python
import jwt
from datetime import datetime, timedelta

SECRET_KEY = 'your-secret-key'

def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
```

#### OAuth 2.0
- Authorization code flow
- Client credentials flow
- Resource owner password credentials
- Implicit flow

### 2. Authorization
- **Role-Based Access Control (RBAC)**
- **Attribute-Based Access Control (ABAC)**
- **Permissions**: Read, Write, Delete, Admin

## API Testing

### 1. Unit Testing
```python
import unittest
from app import app

class APITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_get_users(self):
        response = self.app.get('/api/users')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('users', data)
    
    def test_create_user(self):
        response = self.app.post('/api/users',
                                json={'name': 'Alice', 'email': 'alice@example.com'})
        self.assertEqual(response.status_code, 201)
```

### 2. Integration Testing
- Test full request/response cycle
- Test database interactions
- Test authentication/authorization

### 3. API Testing Tools
- **pytest**: Testing framework
- **requests**: HTTP library for testing
- **httpx**: Async HTTP client
- **Postman**: Manual testing

## Error Handling

### 1. Exception Handling
```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

class ValidationError(Exception):
    pass

@app.errorhandler(ValidationError)
def handle_validation_error(e):
    return jsonify({'error': str(e)}), 400
```

### 2. Error Response Format
```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid input",
        "details": {
            "email": "Invalid email format"
        }
    }
}
```

## API Documentation

### 1. OpenAPI/Swagger
- FastAPI automatically generates
- Flask with flask-swagger-ui
- Describe endpoints, parameters, responses

### 2. Documentation Best Practices
- Clear endpoint descriptions
- Request/response examples
- Authentication requirements
- Error responses

## WebSockets

### 1. Real-Time Communication
```python
from flask_socketio import SocketIO, emit

socketio = SocketIO(app)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('message')
def handle_message(data):
    emit('response', {'data': 'Received: ' + data})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
```

## Caching

### 1. HTTP Caching
- **Cache-Control headers**
- **ETags**
- **Last-Modified**

### 2. Application-Level Caching
```python
from functools import lru_cache
import redis

redis_client = redis.Redis(host='localhost', port=6379)

def get_user_cached(user_id):
    cache_key = f'user:{user_id}'
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    user = get_user_from_db(user_id)
    redis_client.setex(cache_key, 3600, json.dumps(user))
    return user
```

## Rate Limiting

### 1. Implementation
```python
from functools import wraps
from flask import request, jsonify
import time

# Simple in-memory rate limiter
rate_limits = {}

def rate_limit(max_per_minute):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            ip = request.remote_addr
            now = time.time()
            
            if ip not in rate_limits:
                rate_limits[ip] = []
            
            # Remove old requests
            rate_limits[ip] = [t for t in rate_limits[ip] if now - t < 60]
            
            if len(rate_limits[ip]) >= max_per_minute:
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            rate_limits[ip].append(now)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/api/data')
@rate_limit(60)
def get_data():
    return jsonify({'data': 'some data'})
```

## Common Interview Questions

### Q1: Explain the difference between Flask and Django
- **Flask**: Microframework, minimal, flexible, good for APIs
- **Django**: Full-stack, batteries included, ORM, admin, good for web apps

### Q2: What is WSGI?
- Web Server Gateway Interface
- Standard interface between web servers and Python web applications
- Examples: Gunicorn, uWSGI

### Q3: Explain REST principles
- Stateless
- Resource-based URLs
- HTTP methods for actions
- Proper status codes

### Q4: What is the difference between PUT and PATCH?
- **PUT**: Replace entire resource
- **PATCH**: Partial update

### Q5: How do you handle authentication in APIs?
- API keys
- JWT tokens
- OAuth 2.0
- Session-based (for web apps)

### Q6: Explain CORS
- Cross-Origin Resource Sharing
- Allows browsers to make requests to different domains
- Requires proper headers from server

### Q7: What is the difference between synchronous and asynchronous APIs?
- **Synchronous**: Blocking, one request at a time per thread
- **Asynchronous**: Non-blocking, can handle many concurrent requests

### Q8: How do you handle API versioning?
- URL versioning: `/api/v1/users`
- Header versioning: `Accept: application/vnd.api+json;version=1`
- Query parameter: `?version=1`

### Q9: Explain database connection pooling
- Reuse database connections
- Reduces connection overhead
- Limits number of connections
- SQLAlchemy, psycopg2 support pooling

### Q10: How do you test APIs?
- Unit tests for individual functions
- Integration tests for full request/response
- Use test client (Flask, FastAPI provide)
- Mock external dependencies

## Best Practices

### 1. Security
- Input validation
- SQL injection prevention (parameterized queries)
- XSS prevention
- CSRF protection
- HTTPS only
- Rate limiting

### 2. Performance
- Database query optimization
- Caching
- Pagination
- Compression (gzip)
- CDN for static assets

### 3. Code Organization
- Blueprints (Flask) or routers (FastAPI)
- Separate concerns (routes, models, services)
- Configuration management
- Environment variables

### 4. Monitoring
- Logging
- Error tracking (Sentry)
- Performance monitoring
- API analytics

## Common Patterns

### 1. Repository Pattern
```python
class UserRepository:
    def __init__(self, db):
        self.db = db
    
    def get_by_id(self, user_id):
        return self.db.query(User).filter(User.id == user_id).first()
    
    def create(self, user_data):
        user = User(**user_data)
        self.db.add(user)
        self.db.commit()
        return user
```

### 2. Service Layer
```python
class UserService:
    def __init__(self, user_repo):
        self.user_repo = user_repo
    
    def get_user(self, user_id):
        user = self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundError("User not found")
        return user
    
    def create_user(self, user_data):
        # Business logic
        if self.user_repo.email_exists(user_data['email']):
            raise ValidationError("Email already exists")
        return self.user_repo.create(user_data)
```

### 3. Dependency Injection
```python
from fastapi import Depends

def get_user_repository():
    return UserRepository(db)

def get_user_service(repo: UserRepository = Depends(get_user_repository)):
    return UserService(repo)

@app.get("/users/{user_id}")
def get_user(user_id: int, service: UserService = Depends(get_user_service)):
    return service.get_user(user_id)
```
