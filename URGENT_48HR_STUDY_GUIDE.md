# 🚀 URGENT: 48-Hour Interview Prep - Learntube AI + Equip.co Assessment

## ⏰ Quick Study Strategy (48 Hours)

### Day 1 (24 hours): Core Topics
- **Morning (4h)**: Python fundamentals + DSA basics
- **Afternoon (4h)**: FastAPI + SQL
- **Evening (4h)**: Practice coding problems

### Day 2 (24 hours): Practice & Review
- **Morning (4h)**: Mock coding tests
- **Afternoon (4h)**: Review weak areas
- **Evening (4h)**: Final review + rest

---

## 🐍 PYTHON - MUST KNOW (Equip.co Focus)

### 1. Python Fundamentals (High Priority)

#### GIL (Global Interpreter Lock)
```python
# GIL prevents true parallelism in threads
# Only one thread executes Python bytecode at a time
# Use multiprocessing for CPU-bound tasks
# Use threading for I/O-bound tasks
# Use async/await for I/O-bound with better resource usage
```

#### Decorators (Very Common)
```python
# Basic decorator
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function")
        result = func(*args, **kwargs)
        print("After function")
        return result
    return wrapper

@my_decorator
def greet(name):
    return f"Hello, {name}"

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

# Property decorator
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius must be positive")
        self._radius = value
```

#### Generators (Memory Efficient)
```python
# Generator function
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Generator expression
squares = (x**2 for x in range(1000000))  # Memory efficient
squares_list = [x**2 for x in range(1000000)]  # Uses memory

# When to use: Large datasets, streaming data, memory constraints
```

#### Context Managers
```python
# Using with statement
with open('file.txt', 'r') as f:
    content = f.read()

# Custom context manager
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Elapsed: {time.time() - self.start}")

with Timer():
    # Your code
    pass
```

#### List vs Tuple vs Set vs Dict
```python
# List: Mutable, ordered, allows duplicates
lst = [1, 2, 3]

# Tuple: Immutable, ordered, allows duplicates, hashable
tup = (1, 2, 3)

# Set: Mutable, unordered, no duplicates, fast membership test
s = {1, 2, 3}

# Dict: Mutable, ordered (Python 3.7+), key-value pairs
d = {'a': 1, 'b': 2}
```

### 2. Common Python Interview Questions

**Q: What is the difference between `==` and `is`?**
```python
# == checks value equality
# is checks identity (same object in memory)
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)  # True
print(a is b)  # False
```

**Q: Explain `*args` and `**kwargs`**
```python
def func(*args, **kwargs):
    # args is tuple of positional arguments
    # kwargs is dict of keyword arguments
    print(args)    # (1, 2, 3)
    print(kwargs)  # {'a': 4, 'b': 5}

func(1, 2, 3, a=4, b=5)
```

**Q: Mutable default arguments (Common Pitfall)**
```python
# BAD
def append_to(element, target=[]):
    target.append(element)
    return target

# GOOD
def append_to(element, target=None):
    if target is None:
        target = []
    target.append(element)
    return target
```

**Q: List comprehension vs generator**
```python
# List comprehension - creates entire list
[x**2 for x in range(10)]  # [0, 1, 4, 9, ...]

# Generator expression - lazy evaluation
(x**2 for x in range(10))  # Generator object
```

---

## 📊 DATA STRUCTURES & ALGORITHMS (Equip.co Coding Test)

### Must-Know Patterns

#### 1. Two Pointers
```python
def two_sum(nums, target):
    """Find two numbers that add up to target"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

def container_with_most_water(height):
    """Two pointers from both ends"""
    left, right = 0, len(height) - 1
    max_water = 0
    while left < right:
        width = right - left
        area = min(height[left], height[right]) * width
        max_water = max(max_water, area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_water
```

#### 2. Sliding Window
```python
def longest_substring_without_repeating(s):
    """Sliding window with hash map"""
    char_map = {}
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        if char in char_map and char_map[char] >= start:
            start = char_map[char] + 1
        char_map[char] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length

def max_sum_subarray(nums, k):
    """Fixed window size"""
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i-k] + nums[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

#### 3. Binary Search
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def search_rotated_array(nums, target):
    """Search in rotated sorted array"""
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

#### 4. Dynamic Programming
```python
def climb_stairs(n):
    """Fibonacci pattern"""
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

def max_subarray(nums):
    """Kadane's algorithm"""
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

def coin_change(coins, amount):
    """Minimum coins needed"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

#### 5. Stack Problems
```python
def valid_parentheses(s):
    """Stack for matching"""
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    return not stack

def daily_temperatures(temperatures):
    """Monotonic stack"""
    stack = []
    result = [0] * len(temperatures)
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        stack.append(i)
    return result
```

#### 6. Tree Traversal
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_depth(root):
    """DFS"""
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

def level_order(root):
    """BFS"""
    if not root:
        return []
    result = []
    queue = [root]
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.pop(0)
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

### Top 10 DSA Problems for Equip.co

1. **Two Sum** - Hash map
2. **Longest Substring Without Repeating** - Sliding window
3. **Valid Parentheses** - Stack
4. **Merge Two Sorted Lists** - Two pointers
5. **Maximum Subarray** - Kadane's algorithm
6. **Binary Search** - Binary search
7. **Climbing Stairs** - DP (Fibonacci)
8. **Product of Array Except Self** - Prefix/suffix
9. **Group Anagrams** - Hash map
10. **Top K Frequent Elements** - Heap

---

## 🚀 FASTAPI - Essential Knowledge

### 1. Basic FastAPI Setup
```python
from fastapi import FastAPI, HTTPException, Depends, Query, Path
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="API", version="1.0.0")

# Pydantic models
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    age: int

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime

# Basic endpoints
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/users", response_model=List[UserResponse])
async def get_users(skip: int = 0, limit: int = 10):
    # Query parameters
    return []

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int = Path(..., gt=0)):
    # Path parameter with validation
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    return users[user_id]

@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    # Request body
    new_user = UserResponse(
        id=len(users) + 1,
        **user.dict(),
        created_at=datetime.now()
    )
    users.append(new_user)
    return new_user

@app.put("/users/{user_id}")
async def update_user(user_id: int, user: UserCreate):
    # Update logic
    pass

@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: int):
    # Delete logic
    pass
```

### 2. Dependency Injection
```python
from fastapi import Depends

def get_db():
    db = "database_connection"
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme)):
    # Authentication logic
    return user

@app.get("/items")
async def read_items(db = Depends(get_db), user = Depends(get_current_user)):
    return {"db": db, "user": user}
```

### 3. Error Handling
```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "details": exc.errors()}
    )
```

### 4. Async Operations
```python
import asyncio
import aiohttp

@app.get("/fetch-data")
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.example.com/data') as response:
            data = await response.json()
            return data
```

### 5. Database Integration (SQLAlchemy)
```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    email = Column(String(100))

engine = create_engine("sqlite:///./test.db")
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users")
async def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

@app.post("/users")
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
```

### 6. Authentication (JWT)
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username
```

### Common FastAPI Interview Questions

**Q: What is FastAPI?**
- Modern, fast web framework for building APIs
- Based on Starlette and Pydantic
- Automatic API documentation (OpenAPI/Swagger)
- Type hints and validation
- Async support

**Q: How does FastAPI handle async?**
- Built on ASGI (Asynchronous Server Gateway Interface)
- Supports async/await
- Can handle concurrent requests efficiently

**Q: What is Pydantic?**
- Data validation library
- Uses Python type hints
- Automatic validation and serialization
- Used for request/response models

**Q: How do you handle errors in FastAPI?**
- HTTPException for HTTP errors
- Custom exception handlers
- RequestValidationError for validation errors

---

## 🗄️ SQL - Critical Queries

### 1. Basic Queries
```sql
-- SELECT with conditions
SELECT * FROM users WHERE age > 25 AND status = 'active';

-- JOIN
SELECT u.name, o.total
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- LEFT JOIN
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name;

-- Aggregation
SELECT department, AVG(salary) as avg_salary, COUNT(*) as emp_count
FROM employees
GROUP BY department
HAVING COUNT(*) > 10
ORDER BY avg_salary DESC;
```

### 2. Window Functions
```sql
-- ROW_NUMBER
SELECT name, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- RANK vs DENSE_RANK
SELECT name, salary,
       RANK() OVER (ORDER BY salary DESC) as rank,
       DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank
FROM employees;

-- PARTITION BY
SELECT name, department, salary,
       AVG(salary) OVER (PARTITION BY department) as dept_avg
FROM employees;
```

### 3. Common SQL Problems

**Find Second Highest Salary**
```sql
SELECT MAX(salary) as second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Using window function
SELECT salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) as rnk
    FROM employees
) t
WHERE rnk = 2;
```

**Find Duplicate Emails**
```sql
SELECT email, COUNT(*) as count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;
```

**Employees Earning More Than Managers**
```sql
SELECT e.name as employee
FROM employees e
JOIN employees m ON e.manager_id = m.id
WHERE e.salary > m.salary;
```

**Department Highest Salary**
```sql
SELECT d.name as department, e.name as employee, e.salary
FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE (e.department_id, e.salary) IN (
    SELECT department_id, MAX(salary)
    FROM employees
    GROUP BY department_id
);
```

**Consecutive Numbers**
```sql
SELECT DISTINCT l1.num as ConsecutiveNums
FROM logs l1
JOIN logs l2 ON l1.id = l2.id - 1
JOIN logs l3 ON l1.id = l3.id - 2
WHERE l1.num = l2.num AND l2.num = l3.num;
```

### 4. SQLAlchemy Queries
```python
# Basic queries
users = session.query(User).filter(User.age > 25).all()
user = session.query(User).filter(User.id == 1).first()

# Joins
result = session.query(User, Order).join(Order).filter(Order.total > 100).all()

# Aggregation
from sqlalchemy import func
avg_salary = session.query(func.avg(Employee.salary)).scalar()

# Group by
result = session.query(
    Employee.department,
    func.count(Employee.id),
    func.avg(Employee.salary)
).group_by(Employee.department).all()
```

### Common SQL Interview Questions

**Q: Difference between WHERE and HAVING?**
- WHERE filters rows before grouping
- HAVING filters groups after GROUP BY

**Q: Difference between INNER JOIN and LEFT JOIN?**
- INNER JOIN: Only matching rows
- LEFT JOIN: All rows from left table, matching from right (NULL if no match)

**Q: What is an index?**
- Speeds up queries
- Trade-off: Slower writes, more storage
- Use on frequently queried columns

**Q: Explain normalization**
- Organizing data to reduce redundancy
- 1NF: Atomic values
- 2NF: No partial dependencies
- 3NF: No transitive dependencies

---

## 🎯 EQUIP.CO ASSESSMENT TIPS

### Common Test Format
1. **Multiple Choice** - Python concepts, syntax
2. **Coding Problems** - 2-3 problems, 60-90 minutes
3. **System Design** - Sometimes included

### What to Expect
- **Time Limit**: Usually 60-90 minutes
- **Language**: Python only
- **Difficulty**: Medium to Hard
- **Topics**: DSA, Python fundamentals, sometimes system design

### Problem Types
1. **Array/String Manipulation** - Very common
2. **Tree/Graph Problems** - Common
3. **Dynamic Programming** - Sometimes
4. **System Design** - For senior roles

### Tips for Success
1. **Read carefully** - Understand constraints
2. **Start simple** - Brute force first, then optimize
3. **Test edge cases** - Empty arrays, single element, etc.
4. **Time management** - Don't spend too long on one problem
5. **Clean code** - Write readable, well-commented code

### Common Mistakes to Avoid
- ❌ Not handling edge cases
- ❌ Off-by-one errors
- ❌ Not optimizing (O(n²) when O(n) possible)
- ❌ Not testing with examples
- ❌ Poor variable naming

---

## 🏢 LEARNTUBE AI - Company Specific

### Likely Focus Areas
1. **AI/ML Integration** - Python ML libraries
2. **Education Technology** - User management, content delivery
3. **Scalability** - Handle many concurrent users
4. **Data Processing** - Student progress, analytics

### Potential Questions
- How would you design a system to track student progress?
- How would you handle video content delivery at scale?
- How would you implement recommendation algorithms?
- How would you ensure data privacy?

### Tech Stack (Likely)
- **Backend**: FastAPI/Django
- **Database**: PostgreSQL, Redis
- **ML**: scikit-learn, TensorFlow/PyTorch
- **Deployment**: Docker, AWS/GCP
- **Queue**: Celery, RabbitMQ

---

## 📝 QUICK REFERENCE CHEAT SHEET

### Python
```python
# List operations
lst.append(x), lst.pop(), lst.extend([x,y])
lst.sort(), sorted(lst)
lst[::-1]  # Reverse

# Dict operations
d.get(key, default), d.keys(), d.values(), d.items()
d.pop(key), d.update({key: value})

# Set operations
s.add(x), s.remove(x), s.union(t), s.intersection(t)

# String operations
s.split(), s.join(), s.strip(), s.replace(old, new)
s.startswith(prefix), s.endswith(suffix)
```

### Complexity
- O(1): Constant - dict lookup, list append
- O(log n): Logarithmic - binary search
- O(n): Linear - single loop
- O(n log n): Linearithmic - sorting
- O(n²): Quadratic - nested loops
- O(2ⁿ): Exponential - recursive fibonacci

### FastAPI
```python
# Common patterns
@app.get("/items/{item_id}")
async def get_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

# Dependencies
def get_db(): yield db

# Error handling
raise HTTPException(status_code=404, detail="Not found")
```

### SQL
```sql
-- Common patterns
SELECT ... FROM ... WHERE ... GROUP BY ... HAVING ... ORDER BY ... LIMIT ...

-- Window functions
ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)
RANK() OVER (...)
LAG() OVER (...), LEAD() OVER (...)
```

---

## ✅ FINAL CHECKLIST (Before Interview)

- [ ] Review Python fundamentals (decorators, generators, GIL)
- [ ] Practice 10-15 coding problems (focus on arrays, strings, trees)
- [ ] Review FastAPI basics (endpoints, dependencies, error handling)
- [ ] Practice SQL queries (JOINs, window functions, aggregations)
- [ ] Review system design basics (scalability, caching)
- [ ] Prepare 3-5 behavioral stories (STAR method)
- [ ] Research Learntube AI (products, mission, recent news)
- [ ] Test your coding environment
- [ ] Get good sleep!

---

## 🎯 LAST MINUTE TIPS

1. **Stay Calm** - Take deep breaths
2. **Think Out Loud** - Explain your thought process
3. **Ask Questions** - Clarify requirements
4. **Start Simple** - Brute force, then optimize
5. **Test Your Code** - Walk through examples
6. **Time Management** - Don't get stuck on one problem
7. **Clean Code** - Readable, well-named variables

**YOU'VE GOT THIS! 🚀**
