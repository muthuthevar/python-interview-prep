# Python Fundamentals - Interview Questions

## Core Python Concepts

### 1. Python Data Types
- **Mutable vs Immutable Types**
  - Mutable: list, dict, set, bytearray
  - Immutable: int, float, str, tuple, frozenset, bytes
  - Why does this matter? (hashing, thread safety, memory optimization)

- **Type Checking and Type Hints**
  - How to use `typing` module
  - Type hints in function signatures
  - Generic types (List, Dict, Optional, Union)
  - Type checking with mypy

### 2. Memory Management
- **Garbage Collection**
  - Reference counting
  - Generational garbage collection
  - Circular references and how they're handled
  - `gc` module usage

- **Memory Optimization**
  - `__slots__` for memory efficiency
  - Generators vs lists for memory
  - `sys.getsizeof()` and memory profiling

### 3. Python Execution Model
- **Namespaces and Scopes**
  - LEGB rule (Local, Enclosing, Global, Built-in)
  - `global` and `nonlocal` keywords
  - Closure behavior

- **Import System**
  - How imports work
  - `__init__.py` and packages
  - Absolute vs relative imports
  - `sys.path` manipulation
  - Import hooks and meta path finders

### 4. Functions
- **Function Arguments**
  - Positional, keyword, default arguments
  - `*args` and `**kwargs`
  - Argument unpacking
  - Mutable default arguments (common pitfall)

- **First-Class Functions**
  - Functions as objects
  - Higher-order functions
  - Lambda functions (when to use/avoid)
  - Function decorators

- **Decorators**
  - How decorators work
  - Decorator syntax sugar
  - Decorators with arguments
  - Class decorators
  - `functools.wraps` and why it's important

### 5. Comprehensions
- **List, Dict, Set Comprehensions**
  - Syntax and best practices
  - Nested comprehensions
  - When to use vs loops
  - Generator expressions

### 6. Iterators and Generators
- **Iterator Protocol**
  - `__iter__()` and `__next__()`
  - `iter()` and `next()` built-ins
  - StopIteration exception

- **Generators**
  - Generator functions (`yield`)
  - Generator expressions
  - Generator pipelines
  - `yield from` for delegation
  - Memory efficiency of generators

### 7. Context Managers
- **`with` Statement**
  - How context managers work
  - `__enter__()` and `__exit__()`
  - `contextlib` module
  - `@contextmanager` decorator
  - Multiple context managers

### 8. Exception Handling
- **Exception Hierarchy**
  - Built-in exceptions
  - Custom exceptions
  - Exception chaining (`raise ... from`)
  - `else` and `finally` clauses

- **Best Practices**
  - Specific exception catching
  - Exception handling anti-patterns
  - Bare `except:` clause
  - Exception swallowing

### 9. String Operations
- **String Methods**
  - Common string operations
  - String formatting (f-strings, `.format()`, `%`)
  - String immutability
  - String interning

- **Regular Expressions**
  - `re` module
  - Common patterns
  - Performance considerations

### 10. File I/O
- **File Operations**
  - Reading/writing files
  - Binary vs text mode
  - File encoding (UTF-8)
  - `pathlib` module (modern approach)

## Common Interview Questions

### Q1: Explain the difference between `==` and `is`
```python
# == checks value equality
# is checks identity (same object in memory)
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)  # True
print(a is b)  # False
```

### Q2: What is the GIL (Global Interpreter Lock)?
- Prevents multiple native threads from executing Python bytecodes at once
- Impacts CPU-bound multithreading
- Doesn't affect I/O-bound operations
- Solutions: multiprocessing, async/await, C extensions

### Q3: Explain Python's method resolution order (MRO)
- C3 linearization algorithm
- `__mro__` attribute
- Diamond problem in multiple inheritance
- `super()` function behavior

### Q4: What are metaclasses?
- Classes that create classes
- `type` is the default metaclass
- `__new__` and `__init__` in metaclasses
- When to use (rarely, but good to understand)

### Q5: Explain `*args` and `**kwargs`
```python
def func(*args, **kwargs):
    # args is a tuple of positional arguments
    # kwargs is a dict of keyword arguments
    pass
```

### Q6: What is the difference between `__str__` and `__repr__`?
- `__str__`: user-friendly string representation
- `__repr__`: unambiguous representation (ideally valid Python code)
- `repr()` should be unambiguous, `str()` should be readable

### Q7: Explain list slicing
```python
lst = [0, 1, 2, 3, 4, 5]
lst[1:4]    # [1, 2, 3]
lst[::2]    # [0, 2, 4] - step of 2
lst[::-1]   # [5, 4, 3, 2, 1, 0] - reverse
```

### Q8: What are descriptors?
- Objects that define `__get__`, `__set__`, or `__delete__`
- Used for properties, methods, static methods, class methods
- `@property` decorator uses descriptors

### Q9: Explain Python's name mangling
- `__name` becomes `_ClassName__name`
- Used for "private" attributes (convention, not enforcement)
- Prevents accidental overriding in subclasses

### Q10: What is the difference between `__new__` and `__init__`?
- `__new__`: creates the instance (class method)
- `__init__`: initializes the instance (instance method)
- `__new__` is called before `__init__`

## Code Examples to Practice

### Example 1: Mutable Default Arguments
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

### Example 2: List Comprehension vs Generator
```python
# List comprehension - creates entire list in memory
squares = [x**2 for x in range(1000000)]

# Generator expression - lazy evaluation
squares = (x**2 for x in range(1000000))
```

### Example 3: Context Manager Implementation
```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False  # Don't suppress exceptions
```

### Example 4: Decorator with Arguments
```python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")
```
