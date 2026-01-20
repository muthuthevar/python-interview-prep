# Advanced Python Topics - Interview Questions

## Concurrency and Parallelism

### 1. Threading
- **Global Interpreter Lock (GIL)**
  - What it is and why it exists
  - Impact on CPU-bound vs I/O-bound tasks
  - When threading is useful (I/O-bound operations)

- **Threading Module**
  - `threading.Thread`
  - Thread synchronization (locks, semaphores, events)
  - Thread-safe data structures
  - Daemon threads

```python
import threading
import time

def worker(num):
    print(f"Worker {num} starting")
    time.sleep(2)
    print(f"Worker {num} finished")

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

### 2. Multiprocessing
- **When to Use**
  - CPU-bound tasks
  - Bypassing GIL
  - True parallelism

- **Multiprocessing Module**
  - `multiprocessing.Process`
  - Process pools
  - Inter-process communication (Queue, Pipe)
  - Shared memory

```python
import multiprocessing
import time

def cpu_bound_task(n):
    result = sum(i*i for i in range(n))
    return result

if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        results = pool.map(cpu_bound_task, [1000000] * 4)
    print(results)
```

### 3. Async/Await
- **Asynchronous Programming**
  - Event loop
  - Coroutines
  - `async` and `await` keywords
  - `asyncio` module

- **When to Use**
  - I/O-bound operations
  - Network requests
  - Concurrent operations without threads

```python
import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        urls = ['http://example.com'] * 10
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results

asyncio.run(main())
```

### 4. Concurrent.futures
- **ThreadPoolExecutor**
- **ProcessPoolExecutor**
- High-level interface for parallel execution

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def task(n):
    return n * n

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(task, i) for i in range(10)]
    for future in as_completed(futures):
        print(future.result())
```

## Metaclasses and Descriptors

### 1. Metaclasses
- **What are Metaclasses?**
  - Classes that create classes
  - `type` is the default metaclass
  - Advanced topic, rarely needed

```python
class Meta(type):
    def __new__(cls, name, bases, namespace):
        # Modify namespace before class creation
        namespace['created_by'] = 'Meta'
        return super().__new__(cls, name, bases, namespace)

class MyClass(metaclass=Meta):
    pass

print(MyClass.created_by)  # 'Meta'
```

### 2. Descriptors
- **Descriptor Protocol**
  - `__get__`, `__set__`, `__delete__`
  - Used by properties, methods, static methods

```python
class Descriptor:
    def __init__(self, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if value < 0:
            raise ValueError("Value must be positive")
        obj.__dict__[self.name] = value

class MyClass:
    x = Descriptor('x')
    
    def __init__(self, x):
        self.x = x
```

## Decorators (Advanced)

### 1. Decorator Patterns
- **Function Decorators**
- **Class Decorators**
- **Decorators with Arguments**
- **Decorator Chaining**

```python
# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

# Class decorator
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

# Property decorator with caching
def cached_property(func):
    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, '_cache'):
            self._cache = {}
        if func.__name__ not in self._cache:
            self._cache[func.__name__] = func(self)
        return self._cache[func.__name__]
    return wrapper
```

## Generators and Iterators (Advanced)

### 1. Generator Patterns
- **Generator Functions**
- **Generator Expressions**
- **Generator Pipelines**
- **Coroutines with Generators**

```python
# Generator pipeline
def numbers():
    for i in range(10):
        yield i

def square(nums):
    for n in nums:
        yield n * n

def filter_even(nums):
    for n in nums:
        if n % 2 == 0:
            yield n

# Chain generators
result = filter_even(square(numbers()))
list(result)  # [0, 4, 16, 36, 64]
```

### 2. Yield From
- Delegating to sub-generators
- Simplifies generator composition

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

list(combined())  # [1, 2, 3, 4]
```

## Context Managers (Advanced)

### 1. Custom Context Managers
- **Class-based**
- **Function-based with `@contextmanager`**
- **Multiple context managers**

```python
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")

# Usage
with timer():
    # Some operation
    time.sleep(1)
```

### 2. Contextlib Utilities
- `contextlib.suppress` - Suppress exceptions
- `contextlib.redirect_stdout` - Redirect output
- `contextlib.ExitStack` - Multiple context managers

## Memory Management

### 1. Garbage Collection
- **Reference Counting**
- **Generational GC**
- **Circular References**
- **gc Module**

```python
import gc

# Force garbage collection
gc.collect()

# Get GC statistics
gc.get_stats()

# Disable/enable GC
gc.disable()
gc.enable()
```

### 2. Memory Optimization
- **`__slots__`**
- **Generators for memory efficiency**
- **Memory profiling**

```python
class Optimized:
    __slots__ = ['x', 'y']  # Reduces memory footprint
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

## Performance Optimization

### 1. Profiling
- **cProfile**
- **line_profiler**
- **memory_profiler**

```python
import cProfile

def slow_function():
    # Some code
    pass

cProfile.run('slow_function()')
```

### 2. Optimization Techniques
- **Caching (functools.lru_cache)**
- **Cython for speed**
- **NumPy for numerical operations**
- **Just-In-Time compilation (Numba)**

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## Advanced Data Structures

### 1. Collections Module
- **defaultdict**
- **Counter**
- **OrderedDict**
- **deque**
- **ChainMap**
- **namedtuple**

```python
from collections import defaultdict, Counter, deque, namedtuple

# defaultdict
dd = defaultdict(list)
dd['key'].append('value')

# Counter
counter = Counter(['a', 'b', 'a', 'c'])
print(counter.most_common(2))

# deque
dq = deque([1, 2, 3])
dq.appendleft(0)
dq.pop()

# namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
```

### 2. Dataclasses
- **@dataclass decorator**
- **Field customization**
- **Frozen dataclasses**
- **Inheritance**

```python
from dataclasses import dataclass, field

@dataclass
class Person:
    name: str
    age: int
    email: str = field(default='')
    
    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Age cannot be negative")
```

## Type Hints and Static Analysis

### 1. Type Hints
- **Basic types**
- **Generic types (List, Dict, Optional, Union)**
- **Type aliases**
- **Callable types**

```python
from typing import List, Dict, Optional, Union, Callable

def process_items(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}

def find_user(user_id: int) -> Optional[Dict[str, str]]:
    # Returns user dict or None
    pass

def handler(func: Callable[[str], int]) -> None:
    result = func("test")
```

### 2. Type Checking
- **mypy**
- **Type checking in CI/CD**
- **Gradual typing**

## Common Interview Questions

### Q1: Explain the GIL and its implications
- Prevents true parallelism in threads
- Only one thread executes Python bytecode at a time
- Doesn't affect I/O-bound operations
- Solutions: multiprocessing, async/await, C extensions

### Q2: When would you use threading vs multiprocessing vs async?
- **Threading**: I/O-bound tasks, shared state needed
- **Multiprocessing**: CPU-bound tasks, true parallelism needed
- **Async**: I/O-bound tasks, many concurrent operations, single-threaded

### Q3: What are metaclasses and when would you use them?
- Classes that create classes
- Rarely needed in practice
- Used for frameworks (Django ORM, SQLAlchemy)
- Can be used for validation, registration, etc.

### Q4: Explain descriptors
- Objects that define `__get__`, `__set__`, or `__delete__`
- Used by properties, methods, static methods
- Enable computed attributes

### Q5: What is the difference between `yield` and `return`?
- `return`: Exits function, returns value
- `yield`: Pauses function, returns value, can resume
- Functions with `yield` become generators

### Q6: How does Python's garbage collection work?
- Reference counting (primary)
- Generational garbage collection (handles cycles)
- `gc` module for manual control

### Q7: Explain `__slots__`
- Prevents creation of `__dict__`
- Reduces memory usage
- Limits attributes to those specified
- Trade-off: less flexibility

### Q8: What are the differences between `list`, `tuple`, and `array`?
- **list**: Mutable, general purpose
- **tuple**: Immutable, hashable (if elements are)
- **array**: Fixed type, more memory efficient (from `array` module)

### Q9: Explain Python's import system
- Module search path (`sys.path`)
- `__init__.py` files
- Absolute vs relative imports
- Import hooks

### Q10: What is duck typing?
- Focus on behavior, not type
- "If it walks like a duck and quacks like a duck, it's a duck"
- Python's dynamic typing philosophy
- Enables polymorphism without inheritance

## Code Examples

### Example 1: Async Context Manager
```python
import aiohttp
import asyncio

class AsyncSession:
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

async def main():
    async with AsyncSession() as session:
        async with session.get('http://example.com') as response:
            return await response.text()
```

### Example 2: Custom Descriptor
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

class Person:
    name = TypedProperty('name', str)
    age = TypedProperty('age', int)
```

### Example 3: Generator with State
```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Usage
fib = fibonacci()
for _ in range(10):
    print(next(fib))
```
