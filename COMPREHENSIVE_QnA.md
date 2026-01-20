# 📚 COMPREHENSIVE Q&A - EQUIP.CO BACKEND ENGINEER ASSESSMENT

## 🐍 SECTION 1: PYTHON FUNDAMENTALS (50+ Questions)

### Q1: What are the main built-in collection types in Python? Give real backend use cases.

**Answer:**
- **`list`**: Ordered, mutable, allows duplicates
  - Use case: Store user activity logs, maintain order of API requests
  - Example: `user_actions = ['login', 'view_page', 'submit_quiz']`

- **`tuple`**: Ordered, immutable, allows duplicates
  - Use case: Database connection parameters, API endpoint configurations
  - Example: `db_config = ('localhost', 5432, 'learntube')`

- **`set`**: Unordered, mutable, no duplicates, fast membership test
  - Use case: Track unique user IDs, remove duplicates, fast lookups
  - Example: `active_user_ids = {123, 456, 789}`

- **`dict`**: Key-value pairs, ordered (Python 3.7+), fast lookups
  - Use case: User sessions, caching, configuration, API responses
  - Example: `user_session = {'user_id': 123, 'token': 'abc', 'expires': 3600}`

---

### Q2: Explain how `dict` works internally (hashing, average time complexity).

**Answer:**
- **Internal Structure**: Hash table implementation
- **Hashing**: Keys are hashed using `hash()` function to get index
- **Collision Resolution**: Open addressing or chaining
- **Time Complexity**:
  - Average: O(1) for lookup, insert, delete
  - Worst case: O(n) if all keys hash to same bucket
- **Key Requirements**: Keys must be hashable (immutable types)

```python
# Hashable types: int, str, tuple, frozenset
# Non-hashable: list, dict, set

valid_key = (1, 2, 3)  # ✅ Hashable
invalid_key = [1, 2, 3]  # ❌ Not hashable
```

---

### Q3: Show 3 ways to reverse a list or string in Python.

**Answer:**
```python
# Method 1: Slicing (most Pythonic)
lst = [1, 2, 3, 4]
reversed_lst = lst[::-1]  # [4, 3, 2, 1]

# Method 2: Built-in reversed() function
reversed_lst = list(reversed(lst))

# Method 3: In-place reverse
lst.reverse()  # Modifies original list

# For strings
s = "hello"
reversed_s = s[::-1]  # "olleh"
reversed_s = ''.join(reversed(s))
```

---

### Q4: What is list comprehension? Convert a for-loop to comprehension.

**Answer:**
List comprehension is a concise way to create lists.

```python
# For-loop version
squares = []
for x in range(10):
    squares.append(x ** 2)

# List comprehension
squares = [x ** 2 for x in range(10)]

# With condition
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]

# Nested comprehension
matrix = [[i * j for j in range(3)] for i in range(3)]
```

---

### Q5: Show a dict comprehension that inverts mapping and handle duplicates.

**Answer:**
```python
# Original mapping
original = {'a': 1, 'b': 2, 'c': 1}  # Note: duplicate values

# Simple inversion (loses duplicates)
inverted = {v: k for k, v in original.items()}
# Result: {1: 'c', 2: 'b'}  # 'a' is lost!

# Handle duplicates - keep list of keys
from collections import defaultdict
inverted = defaultdict(list)
for k, v in original.items():
    inverted[v].append(k)
# Result: {1: ['a', 'c'], 2: ['b']}
```

---

### Q6: Explain `*args` and `**kwargs` with practical examples.

**Answer:**
```python
def process_data(*args, **kwargs):
    """
    *args: Variable positional arguments (tuple)
    **kwargs: Variable keyword arguments (dict)
    """
    print(f"Positional args: {args}")  # (1, 2, 3)
    print(f"Keyword args: {kwargs}")    # {'name': 'John', 'age': 30}

# Usage
process_data(1, 2, 3, name='John', age=30)

# Practical backend example
def log_event(event_type, *args, user_id=None, timestamp=None, **metadata):
    """Log event with flexible parameters"""
    print(f"Event: {event_type}")
    print(f"Args: {args}")
    print(f"User: {user_id}")
    print(f"Metadata: {metadata}")

log_event('quiz_submitted', 'quiz_123', 'score_85', 
          user_id=123, timestamp='2024-01-01', ip='192.168.1.1')
```

---

### Q7: Explain LEGB with code showing variable name clash.

**Answer:**
LEGB = Local, Enclosing, Global, Built-in (search order)

```python
x = "global"

def outer():
    x = "enclosing"
    
    def inner():
        x = "local"
        print(x)  # "local" - Local scope
    
    inner()
    print(x)  # "enclosing" - Enclosing scope

outer()
print(x)  # "global" - Global scope

# Using nonlocal to modify enclosing scope
def counter():
    count = 0
    def increment():
        nonlocal count  # Modify enclosing scope
        count += 1
        return count
    return increment

# Using global to modify global scope
count = 0
def increment_global():
    global count
    count += 1
```

---

### Q8: Why is `def f(x=[])` a bug? Show fixed version.

**Answer:**
Mutable default arguments are shared across all function calls!

```python
# BUG: Mutable default argument
def append_item(item, target=[]):
    target.append(item)
    return target

print(append_item(1))  # [1]
print(append_item(2))  # [1, 2] - BUG! Should be [2]

# FIX: Use None as default
def append_item(item, target=None):
    if target is None:
        target = []
    target.append(item)
    return target

print(append_item(1))  # [1]
print(append_item(2))  # [2] - Correct!
```

---

### Q9: Difference between `is` and `==` with surprising cases.

**Answer:**
- `==` checks value equality
- `is` checks identity (same object in memory)

```python
# Small integers are interned (cached)
a = 256
b = 256
print(a is b)  # True (same object)
print(a == b)  # True (same value)

# But not always!
a = 257
b = 257
print(a is b)  # False (different objects)
print(a == b)  # True (same value)

# Strings are also interned
a = "hello"
b = "hello"
print(a is b)  # True

# But not if created differently
a = "hello"
b = "".join(['h', 'e', 'l', 'l', 'o'])
print(a is b)  # False
print(a == b)  # True

# Lists are never the same object
a = [1, 2, 3]
b = [1, 2, 3]
print(a is b)  # False
print(a == b)  # True
```

---

### Q10: How does Python handle memory management and garbage collection?

**Answer:**
- **Reference Counting**: Primary method - object deleted when ref count = 0
- **Generational GC**: Handles circular references (3 generations)
- **gc Module**: Manual control with `gc.collect()`

```python
import sys
import gc

# Reference counting
a = [1, 2, 3]
print(sys.getrefcount(a))  # Number of references

# Circular reference
class Node:
    def __init__(self):
        self.ref = None

a = Node()
b = Node()
a.ref = b
b.ref = a  # Circular reference

del a, b
gc.collect()  # Manually trigger GC
```

---

### Q11: What is a generator? Write infinite even numbers generator.

**Answer:**
Generators are memory-efficient iterators using `yield`.

```python
# Generator function
def even_numbers():
    n = 0
    while True:
        yield n
        n += 2

# Use with itertools.islice
from itertools import islice
first_5 = list(islice(even_numbers(), 5))
print(first_5)  # [0, 2, 4, 6, 8]

# Generator expression
squares = (x**2 for x in range(1000000))  # Memory efficient
```

---

### Q12: Difference between generator function and normal function.

**Answer:**
```python
# Normal function - returns all values at once
def squares_list(n):
    result = []
    for i in range(n):
        result.append(i**2)
    return result  # Returns complete list

# Generator function - yields one value at a time
def squares_gen(n):
    for i in range(n):
        yield i**2  # Yields one value, pauses

# Usage
list_squares = squares_list(1000000)  # Uses memory
gen_squares = squares_gen(1000000)    # Memory efficient

# Generator is lazy - only computes when needed
for square in gen_squares:
    if square > 100:
        break  # Only computed up to 10
```

---

### Q13: Implement context manager for file/DB connection.

**Answer:**
```python
# Using class
class DatabaseConnection:
    def __enter__(self):
        print("Opening connection")
        self.conn = "connection_object"
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing connection")
        # Handle exceptions
        return False  # Don't suppress exceptions

# Usage
with DatabaseConnection() as conn:
    # Use connection
    pass

# Using contextlib
from contextlib import contextmanager

@contextmanager
def database_connection():
    print("Opening connection")
    conn = "connection_object"
    try:
        yield conn
    finally:
        print("Closing connection")

# Usage
with database_connection() as conn:
    # Use connection
    pass
```

---

### Q14: Catch specific exception, log it, and re-raise.

**Answer:**
```python
import logging

logger = logging.getLogger(__name__)

try:
    result = 10 / 0
except ZeroDivisionError as e:
    logger.error(f"Division by zero: {e}", exc_info=True)
    raise  # Re-raise same exception

# Or raise different exception
try:
    result = int("abc")
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise TypeError("Expected integer") from e  # Chain exceptions
```

---

### Q15: Write User class with attributes, method, and `__repr__`.

**Answer:**
```python
class User:
    def __init__(self, user_id, email, is_active=True):
        self.user_id = user_id
        self.email = email
        self.is_active = is_active
    
    def activate(self):
        self.is_active = True
    
    def deactivate(self):
        self.is_active = False
    
    def __repr__(self):
        return f"User(user_id={self.user_id}, email='{self.email}', is_active={self.is_active})"
    
    def __str__(self):
        return f"User {self.user_id}: {self.email}"

# Usage
user = User(123, "user@example.com")
print(user)  # Uses __str__
print(repr(user))  # Uses __repr__
```

---

### Q16: Explain `@staticmethod` vs `@classmethod` vs instance methods.

**Answer:**
```python
class User:
    total_users = 0
    
    def __init__(self, name):
        self.name = name
        User.total_users += 1
    
    # Instance method - receives instance as first arg
    def get_name(self):
        return self.name
    
    # Class method - receives class as first arg
    @classmethod
    def get_total_users(cls):
        return cls.total_users
    
    @classmethod
    def create_user(cls, name):
        return cls(name)
    
    # Static method - no implicit first arg
    @staticmethod
    def is_valid_email(email):
        return '@' in email

# Usage
user = User("John")
user.get_name()  # Instance method
User.get_total_users()  # Class method
User.is_valid_email("test@example.com")  # Static method
```

---

### Q17: What is GIL? Impact on CPU-bound vs IO-bound code?

**Answer:**
- **GIL (Global Interpreter Lock)**: Prevents multiple threads from executing Python bytecode simultaneously
- **CPU-bound**: Threading doesn't help (GIL blocks), use multiprocessing
- **IO-bound**: Threading works (GIL released during IO), async/await is better

```python
# CPU-bound: Use multiprocessing
import multiprocessing

def cpu_task(n):
    return sum(i*i for i in range(n))

# Parallel processing
with multiprocessing.Pool() as pool:
    results = pool.map(cpu_task, [1000000]*4)

# IO-bound: Use async/await
import asyncio
import aiohttp

async def fetch_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Concurrent IO
urls = ['http://example.com'] * 10
results = await asyncio.gather(*[fetch_url(url) for url in urls])
```

---

### Q18: How to create and use virtual environment?

**Answer:**
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install packages
pip install fastapi uvicorn

# Freeze requirements
pip freeze > requirements.txt

# Deactivate
deactivate
```

**Why important:**
- Isolate project dependencies
- Avoid version conflicts
- Reproducible environments
- Easy deployment

---

### Q19: Structure a Python package for a service.

**Answer:**
```
my_service/
├── __init__.py
├── main.py
├── config.py
├── models/
│   ├── __init__.py
│   └── user.py
├── routers/
│   ├── __init__.py
│   └── users.py
├── services/
│   ├── __init__.py
│   └── user_service.py
├── database/
│   ├── __init__.py
│   └── connection.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
└── requirements.txt
```

---

### Q20: Additional Python Questions

**Q: What are decorators? Show practical example.**
```python
def timing_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)

slow_function()  # Prints execution time
```

**Q: What is `__slots__`?**
```python
class User:
    __slots__ = ['user_id', 'email']  # Prevents __dict__ creation
    # Reduces memory usage, limits attributes
```

**Q: Explain `__new__` vs `__init__`.**
```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Called after __new__
        pass
```

---

## 📊 SECTION 2: DSA / CODING (40+ Questions)

### Q21: Two Sum - Return indices.

**Answer:**
```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Test
print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
```

---

### Q22: Best time to buy/sell stock - Max profit.

**Answer:**
```python
def max_profit(prices):
    if not prices:
        return 0
    
    min_price = prices[0]
    max_profit = 0
    
    for price in prices[1:]:
        max_profit = max(max_profit, price - min_price)
        min_price = min(min_price, price)
    
    return max_profit

# Test
print(max_profit([7, 1, 5, 3, 6, 4]))  # 5
```

---

### Q23: Kadane's algorithm - Maximum subarray sum.

**Answer:**
```python
def max_subarray(nums):
    max_sum = current_sum = nums[0]
    
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Test
print(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  # 6
```

---

### Q24: Merge two sorted arrays.

**Answer:**
```python
def merge_sorted_arrays(arr1, arr2):
    result = []
    i, j = 0, 0
    
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result

# Test
print(merge_sorted_arrays([1, 3, 5], [2, 4, 6]))  # [1, 2, 3, 4, 5, 6]
```

---

### Q25: Move all zeros to end while maintaining order.

**Answer:**
```python
def move_zeros(nums):
    write_idx = 0
    
    # Move non-zeros to front
    for num in nums:
        if num != 0:
            nums[write_idx] = num
            write_idx += 1
    
    # Fill rest with zeros
    while write_idx < len(nums):
        nums[write_idx] = 0
        write_idx += 1
    
    return nums

# Test
print(move_zeros([0, 1, 0, 3, 12]))  # [1, 3, 12, 0, 0]
```

---

### Q26: Rotate array by k steps (in-place).

**Answer:**
```python
def rotate_array(nums, k):
    n = len(nums)
    k = k % n
    
    # Reverse entire array
    nums.reverse()
    # Reverse first k elements
    nums[:k] = reversed(nums[:k])
    # Reverse remaining elements
    nums[k:] = reversed(nums[k:])
    
    return nums

# Test
print(rotate_array([1, 2, 3, 4, 5], 2))  # [4, 5, 1, 2, 3]
```

---

### Q27: Check if two strings are anagrams.

**Answer:**
```python
def is_anagram(s, t):
    # Normalize: lowercase, remove spaces
    s = s.lower().replace(' ', '')
    t = t.lower().replace(' ', '')
    
    if len(s) != len(t):
        return False
    
    from collections import Counter
    return Counter(s) == Counter(t)

# Test
print(is_anagram("listen", "silent"))  # True
print(is_anagram("The Morse Code", "Here come dots"))  # True
```

---

### Q28: Group anagrams from list of words.

**Answer:**
```python
def group_anagrams(strs):
    from collections import defaultdict
    
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s.lower()))
        groups[key].append(s)
    
    return list(groups.values())

# Test
print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

---

### Q29: Longest common prefix among strings.

**Answer:**
```python
def longest_common_prefix(strs):
    if not strs:
        return ""
    
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix

# Test
print(longest_common_prefix(["flower", "flow", "flight"]))  # "fl"
```

---

### Q30: Remove all occurrences of element in-place, return new length.

**Answer:**
```python
def remove_element(nums, val):
    write_idx = 0
    for read_idx in range(len(nums)):
        if nums[read_idx] != val:
            nums[write_idx] = nums[read_idx]
            write_idx += 1
    return write_idx

# Test
nums = [3, 2, 2, 3]
length = remove_element(nums, 3)
print(length, nums[:length])  # 2 [2, 2]
```

---

### Q31: First non-repeating character in string.

**Answer:**
```python
def first_non_repeating(s):
    from collections import Counter
    
    count = Counter(s)
    for char in s:
        if count[char] == 1:
            return char
    return None

# Test
print(first_non_repeating("leetcode"))  # 'l'
```

---

### Q32: Top-k frequent elements.

**Answer:**
```python
def top_k_frequent(nums, k):
    from collections import Counter
    import heapq
    
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

# Test
print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))  # [1, 2]
```

---

### Q33: Check if permutation of string can form palindrome.

**Answer:**
```python
def can_form_palindrome(s):
    from collections import Counter
    
    count = Counter(s)
    odd_count = sum(1 for c in count.values() if c % 2 == 1)
    return odd_count <= 1

# Test
print(can_form_palindrome("aab"))  # True ("aba")
print(can_form_palindrome("code"))  # False
```

---

### Q34: Find intersection of two arrays (unique elements).

**Answer:**
```python
def intersection(nums1, nums2):
    return list(set(nums1) & set(nums2))

# Test
print(intersection([1, 2, 2, 1], [2, 2]))  # [2]
```

---

### Q35: Longest substring without repeating characters.

**Answer:**
```python
def length_of_longest_substring(s):
    char_map = {}
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        if char in char_map and char_map[char] >= start:
            start = char_map[char] + 1
        char_map[char] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length

# Test
print(length_of_longest_substring("abcabcbb"))  # 3
```

---

### Q36: Longest substring with at most k distinct characters.

**Answer:**
```python
def longest_substring_k_distinct(s, k):
    char_count = {}
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        char_count[char] = char_count.get(char, 0) + 1
        
        while len(char_count) > k:
            char_count[s[start]] -= 1
            if char_count[s[start]] == 0:
                del char_count[s[start]]
            start += 1
        
        max_length = max(max_length, end - start + 1)
    
    return max_length

# Test
print(longest_substring_k_distinct("eceba", 2))  # 3 ("ece")
```

---

### Q37: Smallest subarray with sum ≥ target.

**Answer:**
```python
def min_subarray_len(target, nums):
    min_length = float('inf')
    window_sum = 0
    start = 0
    
    for end in range(len(nums)):
        window_sum += nums[end]
        
        while window_sum >= target:
            min_length = min(min_length, end - start + 1)
            window_sum -= nums[start]
            start += 1
    
    return min_length if min_length != float('inf') else 0

# Test
print(min_subarray_len(7, [2, 3, 1, 2, 4, 3]))  # 2
```

---

### Q38: Reverse singly linked list (iterative and recursive).

**Answer:**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Iterative
def reverse_list_iterative(head):
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    return prev

# Recursive
def reverse_list_recursive(head):
    if not head or not head.next:
        return head
    
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

---

### Q39: Detect cycle in linked list (Floyd's algorithm).

**Answer:**
```python
def has_cycle(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    
    return False
```

---

### Q40: Merge two sorted linked lists.

**Answer:**
```python
def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    current.next = l1 or l2
    return dummy.next
```

---

### Q41: Validate parentheses string using stack.

**Answer:**
```python
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return not stack

# Test
print(is_valid_parentheses("()[]{}"))  # True
print(is_valid_parentheses("([)]"))    # False
```

---

## 🚀 SECTION 3: FASTAPI / PYTHON WEB (30+ Questions)

### Q42: What is FastAPI? Advantages over Flask/Django REST?

**Answer:**
- **FastAPI**: Modern, fast web framework for building APIs
- **Advantages**:
  1. **Performance**: One of the fastest Python frameworks (comparable to Node.js)
  2. **Type Hints**: Automatic validation with Pydantic
  3. **Auto Documentation**: OpenAPI/Swagger UI automatically generated
  4. **Async Support**: Built-in async/await support
  5. **Modern Python**: Uses Python 3.6+ features

```python
# FastAPI example
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

---

### Q43: Write minimal FastAPI app with GET /health endpoint.

**Answer:**
```python
from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }
```

---

### Q44: Show path parameter and query parameter.

**Answer:**
```python
from fastapi import FastAPI, Query

app = FastAPI()

# Path parameter
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}

# Query parameters
@app.get("/items")
async def get_items(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: str = Query(None)
):
    return {
        "limit": limit,
        "offset": offset,
        "search": search
    }
```

---

### Q45: Show Pydantic model for User and use as request body.

**Answer:**
```python
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

app = FastAPI()

class User(BaseModel):
    id: Optional[int] = None
    email: EmailStr
    is_active: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "is_active": True
            }
        }

@app.post("/users", status_code=201)
async def create_user(user: User):
    # user.email is validated automatically
    # user.is_active defaults to True
    return {"message": "User created", "user": user}
```

---

### Q46: How does FastAPI use type hints + Pydantic for validation?

**Answer:**
- FastAPI uses type hints to automatically validate request data
- Pydantic models provide additional validation rules
- Invalid data returns 422 Unprocessable Entity automatically

```python
from pydantic import BaseModel, validator

class User(BaseModel):
    email: str
    age: int = Field(..., gt=0, le=120)
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v

# FastAPI automatically validates and returns errors
```

---

### Q47: Return custom status codes and error messages.

**Answer:**
```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if user_id < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID must be positive"
        )
    
    user = find_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )
    
    return user
```

---

### Q48: Explain dependency injection with DB session example.

**Answer:**
```python
from fastapi import Depends
from sqlalchemy.orm import Session

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

---

### Q49: Show middleware that logs request path and time.

**Answer:**
```python
from fastapi import Request
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import logging

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(
            f"{request.method} {request.url.path} - "
            f"{response.status_code} - {process_time:.3f}s"
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        return response

app.add_middleware(LoggingMiddleware)
```

---

### Q50: When to use `async def` in FastAPI endpoints?

**Answer:**
- Use `async def` for I/O-bound operations (DB queries, API calls)
- Use regular `def` for CPU-bound operations
- Don't block inside async endpoints (use `await`)

```python
# Good: Async for I/O
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = await db.fetch_user(user_id)  # Async DB call
    return user

# Bad: Blocking in async
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = db.fetch_user(user_id)  # Blocking! Don't do this
    return user

# Good: Regular def for CPU-bound
@app.get("/compute")
def compute():
    result = sum(i*i for i in range(1000000))  # CPU-bound
    return {"result": result}
```

---

### Q51: Integrate FastAPI with SQLAlchemy (high level).

**Answer:**
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create engine
engine = create_engine("postgresql://user:pass@localhost/db")
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Define model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Use in endpoint
@app.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    return db.query(User).filter(User.id == user_id).first()
```

---

### Q52: Implement JWT authentication in FastAPI (high level).

**Answer:**
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
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

@app.get("/users/me")
async def read_users_me(current_user: int = Depends(get_current_user)):
    return {"user_id": current_user}
```

---

### Q53: Structure medium-sized FastAPI project.

**Answer:**
```
project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── routers/
│   │   ├── __init__.py
│   │   └── users.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── user_service.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── connection.py
│   └── utils/
│       ├── __init__.py
│       └── auth.py
└── requirements.txt
```

---

## 🗄️ SECTION 4: SQL + BACKEND DATA ACCESS (30+ Questions)

### Q54: Get all active users from users table.

**Answer:**
```sql
SELECT * FROM users WHERE status = 'active';
```

---

### Q55: Get users created in last 7 days.

**Answer:**
```sql
SELECT * FROM users 
WHERE created_at >= NOW() - INTERVAL '7 days';

-- Or
SELECT * FROM users 
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days';
```

---

### Q56: Count users by country.

**Answer:**
```sql
SELECT country, COUNT(*) as user_count
FROM users
GROUP BY country
ORDER BY user_count DESC;
```

---

### Q57: Find total and average order value per user.

**Answer:**
```sql
SELECT 
    user_id,
    COUNT(*) as order_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount
FROM orders
GROUP BY user_id;
```

---

### Q58: Get all orders with user info (INNER JOIN).

**Answer:**
```sql
SELECT 
    o.id as order_id,
    o.amount,
    u.name as user_name,
    u.email
FROM orders o
INNER JOIN users u ON o.user_id = u.id;
```

---

### Q59: Get users who have never placed an order.

**Answer:**
```sql
SELECT u.*
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.id IS NULL;
```

---

### Q60: Get countries with more than 100 users.

**Answer:**
```sql
SELECT country, COUNT(*) as user_count
FROM users
GROUP BY country
HAVING COUNT(*) > 100;
```

---

### Q61: Get products with total sales amount > X.

**Answer:**
```sql
SELECT 
    p.id,
    p.name,
    SUM(oi.quantity * oi.price) as total_sales
FROM products p
JOIN order_items oi ON p.id = oi.product_id
GROUP BY p.id, p.name
HAVING SUM(oi.quantity * oi.price) > 1000;
```

---

### Q62: Get first 20 users ordered by created_at (pagination).

**Answer:**
```sql
SELECT * FROM users
ORDER BY created_at DESC
LIMIT 20 OFFSET 0;

-- Page 2
SELECT * FROM users
ORDER BY created_at DESC
LIMIT 20 OFFSET 20;
```

---

### Q63: Deactivate users that haven't logged in in last year.

**Answer:**
```sql
UPDATE users
SET status = 'inactive'
WHERE last_login < NOW() - INTERVAL '1 year';
```

---

### Q64: Delete orders older than certain date.

**Answer:**
```sql
DELETE FROM orders
WHERE created_at < '2023-01-01';
```

---

### Q65: Design schema for Users and Posts (one-to-many).

**Answer:**
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_posts_user_id ON posts(user_id);
```

---

### Q66: Design schema for Orders and Order Items (one-to-many).

**Answer:**
```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE INDEX idx_order_items_order_id ON order_items(order_id);
```

---

### Q67: Explain primary key vs foreign key.

**Answer:**
- **Primary Key**: Uniquely identifies each row in a table
  - Must be unique and NOT NULL
  - Only one per table
  - Example: `id` in users table

- **Foreign Key**: References primary key in another table
  - Establishes relationship between tables
  - Can be NULL
  - Example: `user_id` in orders table references `id` in users table

```sql
-- Primary key
CREATE TABLE users (
    id SERIAL PRIMARY KEY,  -- Primary key
    email VARCHAR(255)
);

-- Foreign key
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id)  -- Foreign key
);
```

---

### Q68: What is an index? When to add indexes?

**Answer:**
- **Index**: Data structure that speeds up queries
- **When to add**:
  - Columns in WHERE clauses
  - Columns in JOIN conditions
  - Columns in ORDER BY
  - Foreign keys

```sql
-- Index on email (frequent lookups)
CREATE INDEX idx_users_email ON users(email);

-- Index on created_at (sorting)
CREATE INDEX idx_users_created_at ON users(created_at);

-- Index on foreign key
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- Composite index
CREATE INDEX idx_orders_user_status ON orders(user_id, status);
```

---

### Q69: Trade-offs of indexing.

**Answer:**
- **Pros**:
  - Faster SELECT queries
  - Faster JOINs
  - Faster ORDER BY

- **Cons**:
  - Slower INSERT/UPDATE/DELETE (index must be updated)
  - Extra storage space
  - Maintenance overhead

**Best Practice**: Index columns that are frequently queried but rarely updated.

---

### Q70: When to use document DB instead of relational?

**Answer:**
Use document DB (MongoDB) when:
- **Flexible schema**: Data structure changes frequently
- **Denormalized data**: Related data stored together
- **Heavy reads**: Read-heavy workloads
- **Horizontal scaling**: Need to scale across multiple servers
- **JSON-like data**: Natural fit for nested structures

Use relational DB when:
- **ACID transactions**: Need strong consistency
- **Complex queries**: JOINs, aggregations
- **Structured data**: Well-defined schema
- **Relationships**: Complex relationships between entities

---

## 🎯 FINAL TIPS FOR EQUIP.CO ASSESSMENT

1. **Time Management**: 45 minutes total
   - 15 min: Multiple choice questions
   - 30 min: Coding problems

2. **Coding Strategy**:
   - Read problem carefully (2 min)
   - Identify pattern (2 min)
   - Write solution (20 min)
   - Test with examples (5 min)

3. **Common Mistakes to Avoid**:
   - Not handling edge cases
   - Off-by-one errors
   - Not optimizing (O(n²) when O(n) possible)
   - Poor variable naming

4. **Key Patterns to Remember**:
   - Hash map for O(1) lookups
   - Two pointers for sorted arrays
   - Sliding window for substrings
   - Stack for matching problems
   - DP for optimization

**Good luck! 🚀**
