# 🚀 ADVANCED Q&A - DEEPER QUESTIONS FOR EQUIP.CO ASSESSMENT

## 🐍 SECTION 1: ADVANCED PYTHON (40+ Questions)

### Q71: Explain Python's method resolution order (MRO) with multiple inheritance.

**Answer:**
```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

# MRO: D -> B -> C -> A -> object
print(D.__mro__)
# (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)

d = D()
print(d.method())  # "B" (first in MRO)
```

---

### Q72: What are descriptors? Show practical example.

**Answer:**
Descriptors are objects that define `__get__`, `__set__`, or `__delete__`.

```python
class TypedProperty:
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type
    
    def __get__(self, obj, objtype=None):
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"Expected {self.expected_type}")
        obj.__dict__[self.name] = value

class User:
    name = TypedProperty('name', str)
    age = TypedProperty('age', int)
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

user = User("John", 30)
user.age = "thirty"  # TypeError!
```

---

### Q73: Explain `__new__` vs `__init__` with singleton pattern.

**Answer:**
```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Called after __new__
        self.value = None

# Usage
s1 = Singleton()
s1.value = "first"
s2 = Singleton()
print(s2.value)  # "first" - same instance!
print(s1 is s2)  # True
```

---

### Q74: What is `__slots__`? When and why to use it?

**Answer:**
```python
# Without __slots__
class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# With __slots__
class UserOptimized:
    __slots__ = ['name', 'age']  # Prevents __dict__ creation
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Benefits:
# - Reduces memory usage (no __dict__)
# - Faster attribute access
# - Prevents adding new attributes

user = UserOptimized("John", 30)
user.email = "john@example.com"  # AttributeError!
```

---

### Q75: Explain metaclasses with practical example.

**Answer:**
```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "connected"

db1 = Database()
db2 = Database()
print(db1 is db2)  # True
```

---

### Q76: What is `yield from`? Show example.

**Answer:**
```python
def generator1():
    yield 1
    yield 2

def generator2():
    yield 3
    yield 4

def combined():
    yield from generator1()
    yield from generator2()

print(list(combined()))  # [1, 2, 3, 4]

# Practical: Flatten nested lists
def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

print(list(flatten([1, [2, 3], [4, [5, 6]]])))  # [1, 2, 3, 4, 5, 6]
```

---

### Q77: Explain `functools` module - `lru_cache`, `wraps`, `partial`.

**Answer:**
```python
from functools import lru_cache, wraps, partial

# lru_cache - Memoization
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# wraps - Preserve function metadata
def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# partial - Partial function application
def multiply(x, y):
    return x * y

double = partial(multiply, 2)
print(double(5))  # 10
```

---

### Q78: What is `collections` module? Show `defaultdict`, `Counter`, `deque`.

**Answer:**
```python
from collections import defaultdict, Counter, deque

# defaultdict - Default values for missing keys
dd = defaultdict(list)
dd['users'].append('John')
dd['users'].append('Jane')
print(dd['posts'])  # [] (empty list, not KeyError)

# Counter - Count occurrences
counter = Counter(['a', 'b', 'a', 'c', 'b', 'a'])
print(counter)  # Counter({'a': 3, 'b': 2, 'c': 1})
print(counter.most_common(2))  # [('a', 3), ('b', 2)]

# deque - Double-ended queue
dq = deque([1, 2, 3])
dq.appendleft(0)  # O(1)
dq.append(4)      # O(1)
dq.popleft()      # O(1)
print(dq)  # deque([1, 2, 3, 4])
```

---

### Q79: Explain `itertools` - `chain`, `combinations`, `permutations`, `groupby`.

**Answer:**
```python
from itertools import chain, combinations, permutations, groupby

# chain - Combine iterables
list(chain([1, 2], [3, 4], [5, 6]))  # [1, 2, 3, 4, 5, 6]

# combinations - All combinations
list(combinations([1, 2, 3], 2))  # [(1, 2), (1, 3), (2, 3)]

# permutations - All permutations
list(permutations([1, 2, 3], 2))  # [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

# groupby - Group consecutive elements
data = [1, 1, 2, 2, 3, 3, 3]
for key, group in groupby(data):
    print(key, list(group))
# 1 [1, 1]
# 2 [2, 2]
# 3 [3, 3, 3]
```

---

### Q80: What is `dataclasses`? Show example with `@dataclass`.

**Answer:**
```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class User:
    name: str
    age: int
    email: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Age cannot be negative")

user = User("John", 30, "john@example.com")
print(user)  # User(name='John', age=30, email='john@example.com', tags=[])
```

---

### Q81: Explain `typing` module - `Optional`, `Union`, `List`, `Dict`, `Callable`.

**Answer:**
```python
from typing import Optional, Union, List, Dict, Callable

# Optional - Can be None
def get_user(user_id: int) -> Optional[Dict[str, str]]:
    if user_id > 0:
        return {"id": user_id, "name": "John"}
    return None

# Union - Multiple types
def process(value: Union[int, str]) -> str:
    return str(value)

# List, Dict - Generic types
def process_users(users: List[Dict[str, str]]) -> List[str]:
    return [user['name'] for user in users]

# Callable - Function type
def apply_func(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)
```

---

### Q82: What is `contextlib`? Show `@contextmanager` and `ExitStack`.

**Answer:**
```python
from contextlib import contextmanager, ExitStack
import time

@contextmanager
def timer():
    start = time.time()
    try:
        yield
    finally:
        print(f"Elapsed: {time.time() - start:.2f}s")

with timer():
    time.sleep(1)

# ExitStack - Multiple context managers
with ExitStack() as stack:
    file1 = stack.enter_context(open('file1.txt'))
    file2 = stack.enter_context(open('file2.txt'))
    # Both files closed automatically
```

---

### Q83: Explain `asyncio` - `gather`, `create_task`, `wait_for`.

**Answer:**
```python
import asyncio

async def fetch_data(url):
    await asyncio.sleep(1)  # Simulate I/O
    return f"Data from {url}"

# gather - Run concurrently
async def main():
    results = await asyncio.gather(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3")
    )
    return results

# create_task - Schedule coroutine
async def main2():
    task1 = asyncio.create_task(fetch_data("url1"))
    task2 = asyncio.create_task(fetch_data("url2"))
    result1 = await task1
    result2 = await task2

# wait_for - Timeout
async def main3():
    try:
        result = await asyncio.wait_for(fetch_data("url1"), timeout=0.5)
    except asyncio.TimeoutError:
        print("Timeout!")
```

---

### Q84: What is `__call__`? Show callable class example.

**Answer:**
```python
class Counter:
    def __init__(self):
        self.count = 0
    
    def __call__(self):
        self.count += 1
        return self.count

counter = Counter()
print(counter())  # 1
print(counter())  # 2
print(callable(counter))  # True

# Practical: Function with state
class RateLimiter:
    def __init__(self, max_calls):
        self.max_calls = max_calls
        self.calls = []
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            import time
            now = time.time()
            self.calls = [c for c in self.calls if now - c < 60]
            if len(self.calls) >= self.max_calls:
                raise Exception("Rate limit exceeded")
            self.calls.append(now)
            return func(*args, **kwargs)
        return wrapper
```

---

### Q85: Explain `__getitem__`, `__setitem__`, `__len__` for custom sequences.

**Answer:**
```python
class CustomList:
    def __init__(self, items):
        self.items = list(items)
    
    def __getitem__(self, index):
        return self.items[index]
    
    def __setitem__(self, index, value):
        self.items[index] = value
    
    def __len__(self):
        return len(self.items)
    
    def __repr__(self):
        return f"CustomList({self.items})"

lst = CustomList([1, 2, 3])
print(lst[0])      # 1
lst[0] = 10        # __setitem__
print(len(lst))    # 3
print(lst[1:3])    # [2, 3] (slicing works!)
```

---

### Q86: What is `__enter__` and `__exit__`? Show custom context manager.

**Answer:**
```python
class DatabaseConnection:
    def __enter__(self):
        print("Opening connection")
        self.conn = "connection_object"
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing connection")
        # Return False to propagate exceptions
        # Return True to suppress exceptions
        return False

# Usage
with DatabaseConnection() as conn:
    # Use connection
    pass
# Connection automatically closed
```

---

### Q87: Explain `__str__` vs `__repr__` with examples.

**Answer:**
```python
class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f"User: {self.name}"
    
    def __repr__(self):
        return f"User(name='{self.name}', age={self.age})"

user = User("John", 30)
print(str(user))    # User: John (user-friendly)
print(repr(user))   # User(name='John', age=30) (developer-friendly)

# Rule: __repr__ should be unambiguous, ideally valid Python code
```

---

### Q88: What is `__hash__`? When is an object hashable?

**Answer:**
```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

p1 = Point(1, 2)
p2 = Point(1, 2)
print(p1 == p2)        # True
print(hash(p1) == hash(p2))  # True

# Can use as dict key
points = {p1: "point1"}
print(points[p2])  # "point1"

# Immutable types are hashable by default
# Mutable types (list, dict, set) are not hashable
```

---

### Q89: Explain `__iter__` and `__next__` for custom iterators.

**Answer:**
```python
class CountDown:
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

# Usage
for num in CountDown(5):
    print(num)  # 5, 4, 3, 2, 1

# Or manually
counter = CountDown(3)
print(next(counter))  # 3
print(next(counter))  # 2
print(next(counter))  # 1
```

---

### Q90: What is `__getattr__` vs `__getattribute__`?

**Answer:**
```python
class DynamicAttributes:
    def __init__(self):
        self.existing = "I exist"
    
    def __getattr__(self, name):
        # Called only if attribute not found
        return f"Attribute '{name}' not found, but I'll create it!"
    
    def __getattribute__(self, name):
        # Called for ALL attribute access
        if name.startswith('_'):
            raise AttributeError("Private attributes not allowed")
        return super().__getattribute__(name)

obj = DynamicAttributes()
print(obj.existing)      # "I exist"
print(obj.missing)       # "Attribute 'missing' not found..."
print(obj._private)      # AttributeError
```

---

## 📊 SECTION 2: ADVANCED DSA (30+ Questions)

### Q91: Implement LRU Cache from scratch.

**Answer:**
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return -1
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove least recently used (first item)
            self.cache.popitem(last=False)

# Usage
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # 1
cache.put(3, 3)     # Evicts key 2
print(cache.get(2))  # -1 (not found)
```

---

### Q92: Implement Trie (Prefix Tree).

**Answer:**
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Usage
trie = Trie()
trie.insert("apple")
print(trie.search("apple"))    # True
print(trie.search("app"))      # False
print(trie.starts_with("app")) # True
```

---

### Q93: Implement Binary Search Tree with insert, search, delete.

**Answer:**
```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        self.root = self._insert(self.root, val)
    
    def _insert(self, node, val):
        if not node:
            return TreeNode(val)
        if val < node.val:
            node.left = self._insert(node.left, val)
        else:
            node.right = self._insert(node.right, val)
        return node
    
    def search(self, val):
        return self._search(self.root, val)
    
    def _search(self, node, val):
        if not node or node.val == val:
            return node
        if val < node.val:
            return self._search(node.left, val)
        return self._search(node.right, val)
    
    def delete(self, val):
        self.root = self._delete(self.root, val)
    
    def _delete(self, node, val):
        if not node:
            return node
        if val < node.val:
            node.left = self._delete(node.left, val)
        elif val > node.val:
            node.right = self._delete(node.right, val)
        else:
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            # Two children: find inorder successor
            min_node = self._min_value(node.right)
            node.val = min_node.val
            node.right = self._delete(node.right, min_node.val)
        return node
    
    def _min_value(self, node):
        while node.left:
            node = node.left
        return node
```

---

### Q94: Implement Graph with BFS and DFS.

**Answer:**
```python
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)  # Undirected
    
    def bfs(self, start):
        visited = set()
        queue = deque([start])
        visited.add(start)
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return result
    
    def dfs(self, start):
        visited = set()
        result = []
        
        def dfs_helper(node):
            visited.add(node)
            result.append(node)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    dfs_helper(neighbor)
        
        dfs_helper(start)
        return result

# Usage
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
print(g.bfs(0))  # [0, 1, 2, 3]
print(g.dfs(0))  # [0, 1, 3, 2]
```

---

### Q95: Implement Dijkstra's algorithm for shortest path.

**Answer:**
```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Usage
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}
print(dijkstra(graph, 'A'))  # {'A': 0, 'B': 1, 'C': 3, 'D': 4}
```

---

### Q96: Implement Union-Find (Disjoint Set) data structure.

**Answer:**
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)

# Usage
uf = UnionFind(5)
uf.union(0, 1)
uf.union(2, 3)
print(uf.connected(0, 1))  # True
print(uf.connected(0, 2))  # False
```

---

### Q97: Implement sliding window maximum.

**Answer:**
```python
from collections import deque

def max_sliding_window(nums, k):
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove indices with smaller values
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result when window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# Usage
print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))  # [3, 3, 5, 5, 6, 7]
```

---

### Q98: Implement merge sort.

**Answer:**
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage
print(merge_sort([3, 1, 4, 1, 5, 9, 2, 6]))  # [1, 1, 2, 3, 4, 5, 6, 9]
```

---

### Q99: Implement quick sort.

**Answer:**
```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# In-place version
def quick_sort_inplace(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pivot_idx = partition(arr, low, high)
        quick_sort_inplace(arr, low, pivot_idx - 1)
        quick_sort_inplace(arr, pivot_idx + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

---

### Q100: Find longest palindromic substring.

**Answer:**
```python
def longest_palindrome(s):
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    longest = ""
    for i in range(len(s)):
        # Odd length palindromes
        palindrome1 = expand_around_center(i, i)
        # Even length palindromes
        palindrome2 = expand_around_center(i, i + 1)
        longest = max(longest, palindrome1, palindrome2, key=len)
    
    return longest

# Usage
print(longest_palindrome("babad"))  # "bab" or "aba"
```

---

## 🚀 SECTION 3: ADVANCED FASTAPI (20+ Questions)

### Q101: How to implement rate limiting in FastAPI?

**Answer:**
```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import redis.asyncio as redis

app = FastAPI()
redis_client = redis.from_url("redis://localhost")

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        key = f"rate_limit:{client_ip}"
        
        current = await redis_client.get(key)
        if current and int(current) >= self.calls:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        pipe = redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, self.period)
        await pipe.execute()
        
        response = await call_next(request)
        return response

app.add_middleware(RateLimitMiddleware, calls=100, period=60)
```

---

### Q102: How to implement caching in FastAPI with Redis?

**Answer:**
```python
from fastapi import FastAPI, Depends
from functools import wraps
import redis.asyncio as redis
import json
import hashlib

app = FastAPI()
redis_client = redis.from_url("redis://localhost")

def cache_response(ttl: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Check cache
            cached = await redis_client.get(key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await redis_client.setex(key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

@app.get("/users/{user_id}")
@cache_response(ttl=300)
async def get_user(user_id: int):
    # Expensive database query
    return {"user_id": user_id, "name": "John"}
```

---

### Q103: How to handle file uploads in FastAPI?

**Answer:**
```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import aiofiles

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save file
    file_path = f"uploads/{file.filename}"
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return {"filename": file.filename, "size": len(content)}

@app.get("/download/{filename}")
async def download_file(filename: str):
    return FileResponse(f"uploads/{filename}")
```

---

### Q104: How to implement WebSockets in FastAPI?

**Answer:**
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"Message: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

---

### Q105: How to implement background tasks in FastAPI?

**Answer:**
```python
from fastapi import FastAPI, BackgroundTasks
import asyncio

app = FastAPI()

async def send_email(email: str, message: str):
    # Simulate email sending
    await asyncio.sleep(2)
    print(f"Email sent to {email}: {message}")

async def log_event(event: str):
    # Simulate logging
    await asyncio.sleep(1)
    print(f"Event logged: {event}")

@app.post("/users")
async def create_user(
    email: str,
    background_tasks: BackgroundTasks
):
    # Fast response
    user = {"email": email, "id": 1}
    
    # Background tasks (non-blocking)
    background_tasks.add_task(send_email, email, "Welcome!")
    background_tasks.add_task(log_event, f"User created: {email}")
    
    return user
```

---

### Q106: How to implement API versioning in FastAPI?

**Answer:**
```python
from fastapi import FastAPI, APIRouter

app = FastAPI()

# Version 1
v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

@v1_router.get("/users")
async def get_users_v1():
    return {"version": "v1", "users": []}

# Version 2
v2_router = APIRouter(prefix="/api/v2", tags=["v2"])

@v2_router.get("/users")
async def get_users_v2():
    return {"version": "v2", "users": [], "metadata": {}}

app.include_router(v1_router)
app.include_router(v2_router)
```

---

### Q107: How to implement request/response logging in FastAPI?

**Answer:**
```python
from fastapi import FastAPI, Request
from fastapi.middleware.base import BaseHTTPMiddleware
import logging
import time
import json

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        if request.method == "POST":
            body = await request.body()
            logger.debug(f"Request body: {body.decode()}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} - {process_time:.3f}s"
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        return response

app.add_middleware(LoggingMiddleware)
```

---

### Q108: How to implement custom exception handlers?

**Answer:**
```python
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

class CustomException(Exception):
    def __init__(self, message: str, code: int = 400):
        self.message = message
        self.code = code

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    logger.error(f"Custom error: {exc.message}")
    return JSONResponse(
        status_code=exc.code,
        content={"error": exc.message, "type": "CustomException"}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation failed", "details": exc.errors()}
    )

@app.get("/test")
async def test():
    raise CustomException("Something went wrong", code=500)
```

---

## 🗄️ SECTION 4: ADVANCED SQL (20+ Questions)

### Q109: Write query to find nth highest salary.

**Answer:**
```sql
-- Using window function (best approach)
SELECT salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) as rnk
    FROM employees
) t
WHERE rnk = 3;  -- 3rd highest

-- Using subquery
SELECT MAX(salary)
FROM employees
WHERE salary < (
    SELECT MAX(salary)
    FROM employees
    WHERE salary < (SELECT MAX(salary) FROM employees)
);
```

---

### Q110: Write query to find employees with salary greater than their department average.

**Answer:**
```sql
SELECT e.name, e.salary, e.department_id
FROM employees e
WHERE e.salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE department_id = e.department_id
);

-- Using window function
SELECT name, salary, department_id
FROM (
    SELECT 
        name,
        salary,
        department_id,
        AVG(salary) OVER (PARTITION BY department_id) as dept_avg
    FROM employees
) t
WHERE salary > dept_avg;
```

---

### Q111: Write query to find duplicate emails.

**Answer:**
```sql
-- Find duplicates
SELECT email, COUNT(*) as count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- Get all rows with duplicate emails
SELECT u.*
FROM users u
INNER JOIN (
    SELECT email
    FROM users
    GROUP BY email
    HAVING COUNT(*) > 1
) dup ON u.email = dup.email;
```

---

### Q112: Write query to find consecutive numbers.

**Answer:**
```sql
SELECT DISTINCT l1.num as ConsecutiveNums
FROM logs l1
JOIN logs l2 ON l1.id = l2.id - 1
JOIN logs l3 ON l1.id = l3.id - 2
WHERE l1.num = l2.num AND l2.num = l3.num;
```

---

### Q113: Write query to find department with highest average salary.

**Answer:**
```sql
SELECT department_id, AVG(salary) as avg_salary
FROM employees
GROUP BY department_id
ORDER BY avg_salary DESC
LIMIT 1;

-- Using window function
SELECT department_id, avg_salary
FROM (
    SELECT 
        department_id,
        AVG(salary) as avg_salary,
        RANK() OVER (ORDER BY AVG(salary) DESC) as rnk
    FROM employees
    GROUP BY department_id
) t
WHERE rnk = 1;
```

---

### Q114: Write query to find employees who have never taken a leave.

**Answer:**
```sql
SELECT e.*
FROM employees e
LEFT JOIN leaves l ON e.id = l.employee_id
WHERE l.id IS NULL;
```

---

### Q115: Write query to find top 3 products by sales in each category.

**Answer:**
```sql
SELECT category, product_id, total_sales
FROM (
    SELECT 
        p.category,
        p.id as product_id,
        SUM(oi.quantity * oi.price) as total_sales,
        ROW_NUMBER() OVER (PARTITION BY p.category ORDER BY SUM(oi.quantity * oi.price) DESC) as rnk
    FROM products p
    JOIN order_items oi ON p.id = oi.product_id
    GROUP BY p.category, p.id
) t
WHERE rnk <= 3;
```

---

### Q116: Write query to calculate running total.

**Answer:**
```sql
SELECT 
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) as running_total
FROM transactions
ORDER BY date;
```

---

### Q117: Write query to find employees earning more than their manager.

**Answer:**
```sql
SELECT e.name as employee, e.salary as employee_salary,
       m.name as manager, m.salary as manager_salary
FROM employees e
JOIN employees m ON e.manager_id = m.id
WHERE e.salary > m.salary;
```

---

### Q118: Write query to find customers who bought all products.

**Answer:**
```sql
SELECT customer_id
FROM (
    SELECT DISTINCT customer_id, product_id
    FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
) customer_products
GROUP BY customer_id
HAVING COUNT(DISTINCT product_id) = (
    SELECT COUNT(*) FROM products
);
```

---

## 🎯 FINAL TIPS

1. **Practice Pattern Recognition**: Identify patterns quickly
2. **Time Management**: Allocate time wisely
3. **Edge Cases**: Always consider edge cases
4. **Code Quality**: Write clean, readable code
5. **Explain Your Approach**: Think out loud

**Total Questions: 118+ covering all topics!**

Good luck! 🚀
