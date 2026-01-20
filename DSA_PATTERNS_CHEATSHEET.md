# 🎯 DSA PATTERNS CHEAT SHEET - MEMORIZE THESE!

## ⚡ Quick Pattern Recognition Guide

### 1. TWO POINTERS
**When to use:**
- Sorted array
- Palindrome checking
- Pair sum problems
- Container problems

**Template:**
```python
left, right = 0, len(arr) - 1
while left < right:
    # Process
    if condition:
        left += 1
    else:
        right -= 1
```

**Problems:**
- Two Sum (sorted)
- Three Sum
- Container With Most Water
- Valid Palindrome
- Trapping Rain Water

---

### 2. SLIDING WINDOW
**When to use:**
- Substring/subarray problems
- Fixed or variable window size
- "Longest/shortest substring with condition"

**Template:**
```python
start = 0
for end in range(len(s)):
    # Expand window
    while condition_not_met:
        # Shrink window
        start += 1
    # Update result
```

**Problems:**
- Longest Substring Without Repeating
- Minimum Window Substring
- Maximum Sum Subarray of Size K
- Longest Repeating Character Replacement

---

### 3. HASH MAP / SET
**When to use:**
- Need O(1) lookup
- Counting frequencies
- Finding duplicates
- Two Sum type problems

**Template:**
```python
seen = {}
for item in items:
    if item in seen:
        # Found
    seen[item] = value
```

**Problems:**
- Two Sum
- Group Anagrams
- Contains Duplicate
- Longest Consecutive Sequence

---

### 4. STACK
**When to use:**
- Matching problems (parentheses)
- Next greater/smaller element
- Monotonic stack problems
- Reversing/undoing operations

**Template:**
```python
stack = []
for item in items:
    while stack and condition:
        process(stack.pop())
    stack.append(item)
```

**Problems:**
- Valid Parentheses
- Daily Temperatures
- Largest Rectangle in Histogram
- Next Greater Element

---

### 5. BINARY SEARCH
**When to use:**
- Sorted array
- Finding boundaries
- Search in rotated array
- "Find first/last occurrence"

**Template:**
```python
left, right = 0, len(arr) - 1
while left <= right:
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
```

**Problems:**
- Binary Search
- Search in Rotated Array
- Find First/Last Position
- Search 2D Matrix

---

### 6. DYNAMIC PROGRAMMING
**When to use:**
- Optimization problems
- Overlapping subproblems
- "How many ways" or "minimum/maximum"
- Can break into smaller problems

**Template:**
```python
# 1D DP
dp = [0] * (n + 1)
dp[0] = base_case
for i in range(1, n + 1):
    dp[i] = recurrence_relation

# 2D DP
dp = [[0] * (n + 1) for _ in range(m + 1)]
for i in range(1, m + 1):
    for j in range(1, n + 1):
        dp[i][j] = recurrence_relation
```

**Problems:**
- Climbing Stairs (Fibonacci)
- Maximum Subarray (Kadane's)
- Coin Change
- Longest Common Subsequence
- Edit Distance

---

### 7. TREE TRAVERSAL
**When to use:**
- Tree problems
- Level-order = BFS
- Inorder/Preorder/Postorder = DFS

**Template:**
```python
# DFS (Recursive)
def dfs(node):
    if not node:
        return
    # Process node (preorder)
    dfs(node.left)
    # Process node (inorder)
    dfs(node.right)
    # Process node (postorder)

# BFS (Iterative)
queue = [root]
while queue:
    node = queue.pop(0)
    # Process node
    if node.left:
        queue.append(node.left)
    if node.right:
        queue.append(node.right)
```

**Problems:**
- Maximum Depth
- Same Tree
- Level Order Traversal
- Validate BST
- Lowest Common Ancestor

---

### 8. GRAPH TRAVERSAL
**When to use:**
- Connected components
- Path finding
- Cycle detection

**Template:**
```python
# DFS
def dfs(node, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, visited)

# BFS
queue = [start]
visited = {start}
while queue:
    node = queue.pop(0)
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)
```

**Problems:**
- Number of Islands
- Clone Graph
- Course Schedule
- Word Ladder

---

### 9. HEAP / PRIORITY QUEUE
**When to use:**
- Top K problems
- Merge K sorted lists
- Find median
- Need min/max efficiently

**Template:**
```python
import heapq

# Min heap (default)
heap = []
heapq.heappush(heap, item)
min_item = heapq.heappop(heap)

# Max heap (negate values)
heapq.heappush(heap, -item)
max_item = -heapq.heappop(heap)
```

**Problems:**
- Top K Frequent Elements
- Kth Largest Element
- Merge K Sorted Lists
- Find Median from Data Stream

---

### 10. BACKTRACKING
**When to use:**
- Generate all combinations/permutations
- Constraint satisfaction
- "Find all solutions"

**Template:**
```python
def backtrack(current, ...):
    if is_solution(current):
        result.append(current[:])
        return
    
    for choice in choices:
        if is_valid(choice):
            current.append(choice)
            backtrack(current, ...)
            current.pop()  # Undo
```

**Problems:**
- Generate Parentheses
- Permutations
- Subsets
- N-Queens
- Sudoku Solver

---

## 🎯 PROBLEM → PATTERN MAPPING

| Problem Type | Pattern | Time Complexity |
|-------------|---------|----------------|
| Find pair that sums to target | Hash Map | O(n) |
| Longest substring with condition | Sliding Window | O(n) |
| Valid parentheses | Stack | O(n) |
| Search in sorted array | Binary Search | O(log n) |
| Maximum subarray | DP (Kadane's) | O(n) |
| Ways to climb stairs | DP (Fibonacci) | O(n) |
| Top K elements | Heap | O(n log k) |
| All permutations | Backtracking | O(n!) |
| Number of islands | DFS/BFS | O(m×n) |
| Merge sorted lists | Two Pointers | O(n+m) |

---

## 🚨 COMMON MISTAKES TO AVOID

1. **Off-by-one errors**
   - Check boundaries: `i < len(arr)` vs `i <= len(arr) - 1`
   - Array indices: 0-indexed vs 1-indexed

2. **Not handling edge cases**
   - Empty array/string
   - Single element
   - All same elements
   - Null/None values

3. **Wrong time complexity**
   - Nested loops = O(n²)
   - Sorting = O(n log n)
   - Hash map lookup = O(1) average

4. **Not optimizing**
   - O(n²) when O(n) possible
   - Using list when set/dict better
   - Not using two pointers when applicable

5. **Modifying while iterating**
   - Don't modify list while iterating
   - Use indices or create new list

---

## ⚡ QUICK DECISION TREE

```
Is array sorted?
├─ YES → Binary Search or Two Pointers
└─ NO → Can we sort it?
    ├─ YES → Sort then use sorted techniques
    └─ NO → Hash Map, Sliding Window, or DP

Is it a substring/subarray problem?
├─ YES → Sliding Window
└─ NO → Continue...

Is it optimization (min/max/count ways)?
├─ YES → Dynamic Programming
└─ NO → Continue...

Is it matching/validating?
├─ YES → Stack or Hash Map
└─ NO → Continue...

Is it tree/graph?
├─ YES → DFS or BFS
└─ NO → Continue...

Is it "all combinations"?
├─ YES → Backtracking
└─ NO → Hash Map or other pattern
```

---

## 📝 MEMORIZE THESE TEMPLATES

### Two Sum (Hash Map)
```python
seen = {}
for i, num in enumerate(nums):
    complement = target - num
    if complement in seen:
        return [seen[complement], i]
    seen[num] = i
```

### Sliding Window
```python
start = 0
for end in range(len(s)):
    # Expand
    while invalid:
        # Shrink
        start += 1
    # Update result
```

### Kadane's Algorithm
```python
max_sum = current_sum = nums[0]
for num in nums[1:]:
    current_sum = max(num, current_sum + num)
    max_sum = max(max_sum, current_sum)
```

### Binary Search
```python
left, right = 0, len(nums) - 1
while left <= right:
    mid = (left + right) // 2
    if nums[mid] == target:
        return mid
    elif nums[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
```

---

## 🎯 TOP 20 MUST-SOLVE PROBLEMS

1. ✅ Two Sum
2. ✅ Longest Substring Without Repeating
3. ✅ Valid Parentheses
4. ✅ Maximum Subarray
5. ✅ Binary Search
6. ✅ Climbing Stairs
7. ✅ Merge Two Sorted Lists
8. ✅ Reverse Linked List
9. ✅ Group Anagrams
10. ✅ Product of Array Except Self
11. ✅ Container With Most Water
12. ✅ Coin Change
13. ✅ Top K Frequent Elements
14. ✅ Number of Islands
15. ✅ Validate BST
16. ✅ Course Schedule
17. ✅ Generate Parentheses
18. ✅ Longest Common Subsequence
19. ✅ Search in Rotated Array
20. ✅ Daily Temperatures

**Master these 20 and you'll handle 80% of interview problems!**
