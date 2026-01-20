# Concurrency & Async Programming - Interview Questions

## Threading

### 1. Threading Basics
- **Global Interpreter Lock (GIL)**
  - Only one thread executes Python bytecode at a time
  - Prevents true parallelism for CPU-bound tasks
  - Doesn't block I/O operations

- **When to Use Threading**
  - I/O-bound operations (file I/O, network requests)
  - Tasks that spend time waiting
  - GUI applications (keep UI responsive)

```python
import threading
import time

def worker(num):
    print(f"Worker {num} starting")
    time.sleep(2)  # Simulate I/O operation
    print(f"Worker {num} finished")

# Create threads
threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

print("All threads completed")
```

### 2. Thread Synchronization

#### Locks
```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100000):
        with lock:  # Acquire lock
            counter += 1
        # Lock automatically released

threads = [threading.Thread(target=increment) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Counter: {counter}")  # Should be 500000
```

#### Semaphores
```python
import threading
import time

semaphore = threading.Semaphore(3)  # Allow 3 concurrent accesses

def worker(num):
    with semaphore:
        print(f"Worker {num} acquired semaphore")
        time.sleep(2)
        print(f"Worker {num} releasing semaphore")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

#### Events
```python
import threading
import time

event = threading.Event()

def waiter():
    print("Waiting for event...")
    event.wait()  # Block until event is set
    print("Event received!")

def setter():
    time.sleep(3)
    print("Setting event")
    event.set()

threading.Thread(target=waiter).start()
threading.Thread(target=setter).start()
```

#### Condition Variables
```python
import threading
import time

condition = threading.Condition()
items = []

def consumer():
    with condition:
        while len(items) == 0:
            condition.wait()  # Wait for notification
        item = items.pop(0)
        print(f"Consumed: {item}")

def producer():
    time.sleep(1)
    with condition:
        items.append("item")
        condition.notify()  # Notify waiting threads

threading.Thread(target=consumer).start()
threading.Thread(target=producer).start()
```

### 3. Thread-Safe Data Structures
```python
import queue

# Thread-safe queue
q = queue.Queue()

# Producer
def producer():
    for i in range(10):
        q.put(i)
        print(f"Produced: {i}")

# Consumer
def consumer():
    while True:
        item = q.get()
        if item is None:  # Poison pill
            break
        print(f"Consumed: {item}")
        q.task_done()

threading.Thread(target=producer).start()
threading.Thread(target=consumer).start()
q.join()  # Wait for all tasks to be processed
```

## Multiprocessing

### 1. Multiprocessing Basics
- **Bypasses GIL**
- **True parallelism**
- **Separate memory spaces**
- **Use for CPU-bound tasks**

```python
import multiprocessing
import time

def cpu_bound_task(n):
    result = sum(i * i for i in range(n))
    return result

if __name__ == '__main__':
    # Sequential
    start = time.time()
    results = [cpu_bound_task(1000000) for _ in range(4)]
    print(f"Sequential: {time.time() - start:.2f}s")
    
    # Parallel
    start = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(cpu_bound_task, [1000000] * 4)
    print(f"Parallel: {time.time() - start:.2f}s")
```

### 2. Process Communication

#### Queue
```python
import multiprocessing

def worker(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Processed: {item}")

if __name__ == '__main__':
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()
    
    for i in range(10):
        q.put(i)
    q.put(None)  # Signal to stop
    p.join()
```

#### Pipe
```python
import multiprocessing

def sender(conn):
    conn.send("Hello from sender")
    conn.close()

def receiver(conn):
    msg = conn.recv()
    print(f"Received: {msg}")
    conn.close()

if __name__ == '__main__':
    parent_conn, child_conn = multiprocessing.Pipe()
    
    p1 = multiprocessing.Process(target=sender, args=(child_conn,))
    p2 = multiprocessing.Process(target=receiver, args=(parent_conn,))
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()
```

#### Shared Memory
```python
import multiprocessing

def worker(shared_array):
    for i in range(len(shared_array)):
        shared_array[i] = i * 2

if __name__ == '__main__':
    # Create shared array
    shared_array = multiprocessing.Array('i', 10)
    
    p = multiprocessing.Process(target=worker, args=(shared_array,))
    p.start()
    p.join()
    
    print(list(shared_array))
```

### 3. Process Pool
```python
import multiprocessing

def square(x):
    return x * x

if __name__ == '__main__':
    with multiprocessing.Pool(processes=4) as pool:
        # Map
        results = pool.map(square, range(10))
        print(results)
        
        # Apply async
        result = pool.apply_async(square, (10,))
        print(result.get())
        
        # Map async
        results = pool.map_async(square, range(10))
        print(results.get())
```

## Async/Await

### 1. Async Basics
- **Event loop**
- **Coroutines**
- **Non-blocking I/O**
- **Single-threaded concurrency**

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

# Run
results = asyncio.run(main())
```

### 2. Async Patterns

#### Sequential Execution
```python
import asyncio

async def task1():
    await asyncio.sleep(1)
    return "Task 1"

async def task2():
    await asyncio.sleep(1)
    return "Task 2"

async def main():
    result1 = await task1()
    result2 = await task2()
    return result1, result2
```

#### Concurrent Execution
```python
async def main():
    # Using gather
    results = await asyncio.gather(task1(), task2())
    
    # Using create_task
    task1_coro = asyncio.create_task(task1())
    task2_coro = asyncio.create_task(task2())
    result1 = await task1_coro
    result2 = await task2_coro
```

#### Timeout
```python
async def slow_task():
    await asyncio.sleep(5)
    return "Done"

async def main():
    try:
        result = await asyncio.wait_for(slow_task(), timeout=2.0)
    except asyncio.TimeoutError:
        print("Task timed out")
```

### 3. Async Context Managers
```python
import asyncio

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource")
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource")
        await asyncio.sleep(0.1)

async def main():
    async with AsyncResource() as resource:
        print("Using resource")
        await asyncio.sleep(1)
```

### 4. Async Generators
```python
async def async_generator():
    for i in range(5):
        await asyncio.sleep(0.1)
        yield i

async def main():
    async for value in async_generator():
        print(value)
```

## Concurrent.futures

### 1. ThreadPoolExecutor
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def io_bound_task(n):
    time.sleep(1)  # Simulate I/O
    return n * n

with ThreadPoolExecutor(max_workers=5) as executor:
    # Submit tasks
    futures = [executor.submit(io_bound_task, i) for i in range(10)]
    
    # Get results as they complete
    for future in as_completed(futures):
        result = future.result()
        print(result)
```

### 2. ProcessPoolExecutor
```python
from concurrent.futures import ProcessPoolExecutor

def cpu_bound_task(n):
    return sum(i * i for i in range(n))

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cpu_bound_task, 1000000) for _ in range(4)]
        results = [f.result() for f in futures]
        print(results)
```

## Common Interview Questions

### Q1: Explain the GIL and its impact
- Global Interpreter Lock prevents multiple threads from executing Python bytecode simultaneously
- Impacts CPU-bound multithreading (no true parallelism)
- Doesn't affect I/O-bound operations (threads release GIL during I/O)
- Solutions: multiprocessing, async/await, C extensions

### Q2: When would you use threading vs multiprocessing vs async?
- **Threading**: I/O-bound tasks, shared state needed, simple parallelism
- **Multiprocessing**: CPU-bound tasks, true parallelism needed, separate memory spaces
- **Async**: I/O-bound tasks, many concurrent operations, single-threaded, better resource usage

### Q3: What is a race condition?
- When multiple threads/processes access shared data simultaneously
- Result depends on timing
- Prevented with locks, semaphores, or thread-safe data structures

### Q4: Explain deadlock
- Two or more threads waiting for each other to release resources
- Prevention: acquire locks in same order, use timeout, avoid nested locks

### Q5: What is the difference between `asyncio.gather()` and `asyncio.create_task()`?
- `gather()`: Waits for all coroutines, returns results in order
- `create_task()`: Schedules coroutine, returns Task object, can await individually

### Q6: How does async/await work?
- `async` defines coroutine function
- `await` pauses execution until coroutine completes
- Event loop manages coroutine execution
- Non-blocking I/O operations

### Q7: What is a coroutine?
- Function defined with `async def`
- Returns coroutine object when called
- Can be paused and resumed
- Must be awaited or run in event loop

### Q8: Explain thread safety
- Code that can be safely executed by multiple threads simultaneously
- Achieved through synchronization primitives (locks, semaphores)
- Thread-safe data structures (queue.Queue)

### Q9: What is the difference between `multiprocessing.Queue` and `queue.Queue`?
- `queue.Queue`: Thread-safe, for threading
- `multiprocessing.Queue`: Process-safe, for multiprocessing, uses pickle for serialization

### Q10: How do you handle exceptions in async code?
```python
async def task():
    raise ValueError("Error")

async def main():
    try:
        await task()
    except ValueError as e:
        print(f"Caught: {e}")
    
    # Or with gather
    results = await asyncio.gather(task(), return_exceptions=True)
```

## Best Practices

### 1. Threading
- Use locks for shared mutable state
- Prefer thread-safe data structures when possible
- Use daemon threads for background tasks
- Avoid global state when possible

### 2. Multiprocessing
- Use `if __name__ == '__main__':` guard
- Minimize data sharing (use queues/pipes)
- Consider overhead (process creation is expensive)
- Use process pools for similar tasks

### 3. Async
- Use for I/O-bound operations
- Don't use blocking operations in async code
- Use `asyncio.gather()` for concurrent operations
- Handle exceptions properly
- Use async context managers for resources

## Common Patterns

### 1. Producer-Consumer (Threading)
```python
import queue
import threading

def producer(q):
    for i in range(10):
        q.put(i)
        print(f"Produced: {i}")

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Consumed: {item}")
        q.task_done()

q = queue.Queue()
threading.Thread(target=producer, args=(q,)).start()
threading.Thread(target=consumer, args=(q,)).start()
q.join()
```

### 2. Parallel Processing (Multiprocessing)
```python
import multiprocessing

def process_chunk(chunk):
    return sum(x * x for x in chunk)

if __name__ == '__main__':
    data = list(range(1000000))
    chunk_size = len(data) // 4
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    with multiprocessing.Pool() as pool:
        results = pool.map(process_chunk, chunks)
    
    total = sum(results)
```

### 3. Concurrent HTTP Requests (Async)
```python
import aiohttp
import asyncio

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

urls = ['http://example.com'] * 100
results = asyncio.run(fetch_all(urls))
```

### 4. Rate Limiting (Async)
```python
import asyncio
from collections import deque

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
    
    async def acquire(self):
        now = asyncio.get_event_loop().time()
        # Remove old calls
        while self.calls and self.calls[0] <= now - self.period:
            self.calls.popleft()
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            await asyncio.sleep(sleep_time)
            return await self.acquire()
        
        self.calls.append(now)

limiter = RateLimiter(max_calls=10, period=1.0)

async def limited_task():
    await limiter.acquire()
    # Do work
    pass
```
