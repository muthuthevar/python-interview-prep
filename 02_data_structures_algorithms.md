# Data Structures & Algorithms - Interview Questions

## Data Structures

### 1. Lists
- **Operations and Complexity**
  - Access: O(1)
  - Append: O(1) amortized
  - Insert: O(n)
  - Delete: O(n)
  - Search: O(n)

- **Common Patterns**
  - Two pointers technique
  - Sliding window
  - Prefix sums
  - List rotation

### 2. Dictionaries (Hash Tables)
- **Operations and Complexity**
  - Access: O(1) average, O(n) worst case
  - Insert: O(1) average, O(n) worst case
  - Delete: O(1) average, O(n) worst case
  - Search: O(1) average, O(n) worst case

- **Key Concepts**
  - Hash function and collision resolution
  - Dictionary ordering (Python 3.7+)
  - `defaultdict`, `Counter`, `OrderedDict`
  - Dictionary comprehension

### 3. Sets
- **Operations and Complexity**
  - Add: O(1) average
  - Remove: O(1) average
  - Membership test: O(1) average
  - Union, Intersection: O(n+m)

- **Use Cases**
  - Removing duplicates
  - Fast membership testing
  - Set operations (union, intersection, difference)

### 4. Tuples
- **Characteristics**
  - Immutable
  - Hashable (if all elements are hashable)
  - Memory efficient
  - Used as dictionary keys

### 5. Stacks and Queues
- **Stack Implementation**
  - Using list (append/pop)
  - Using collections.deque
  - LIFO (Last In First Out)

- **Queue Implementation**
  - Using collections.deque
  - Using queue.Queue (thread-safe)
  - FIFO (First In First Out)

- **Priority Queue**
  - Using heapq module
  - Using queue.PriorityQueue

### 6. Trees
- **Binary Tree**
  - Node structure
  - Traversal: Inorder, Preorder, Postorder
  - Level-order traversal (BFS)

- **Binary Search Tree (BST)**
  - Properties
  - Search, Insert, Delete operations
  - Balanced BST (AVL, Red-Black)

- **Trie (Prefix Tree)**
  - Use cases (autocomplete, spell checker)
  - Implementation

### 7. Graphs
- **Representation**
  - Adjacency list
  - Adjacency matrix
  - Edge list

- **Traversal**
  - BFS (Breadth-First Search)
  - DFS (Depth-First Search)

- **Algorithms**
  - Shortest path (Dijkstra, Bellman-Ford)
  - Topological sort
  - Cycle detection

### 8. Heaps
- **Min Heap vs Max Heap**
  - Heap property
  - Operations: insert, extract_min/max, heapify
  - Using heapq module (min heap only)

## Algorithms

### 1. Sorting Algorithms
- **Comparison Sorts**
  - Bubble Sort: O(n²)
  - Selection Sort: O(n²)
  - Insertion Sort: O(n²)
  - Merge Sort: O(n log n)
  - Quick Sort: O(n log n) average, O(n²) worst
  - Heap Sort: O(n log n)

- **Non-Comparison Sorts**
  - Counting Sort: O(n+k)
  - Radix Sort: O(d(n+k))
  - Bucket Sort: O(n+k)

- **Python's Timsort**
  - Hybrid of merge sort and insertion sort
  - Stable and adaptive
  - Used by Python's `sorted()` and `.sort()`

### 2. Searching Algorithms
- **Linear Search**: O(n)
- **Binary Search**: O(log n)
  - Requirements: sorted array
  - Iterative and recursive implementations
  - Finding boundaries (first/last occurrence)

### 3. Dynamic Programming
- **Key Concepts**
  - Overlapping subproblems
  - Optimal substructure
  - Memoization vs tabulation

- **Classic Problems**
  - Fibonacci sequence
  - Longest Common Subsequence
  - Longest Increasing Subsequence
  - Knapsack problem
  - Coin change
  - Edit distance

### 4. Greedy Algorithms
- **Characteristics**
  - Makes locally optimal choice
  - Doesn't always guarantee global optimum

- **Problems**
  - Activity selection
  - Fractional knapsack
  - Minimum spanning tree (Kruskal, Prim)
  - Dijkstra's shortest path

### 5. Backtracking
- **Concept**
  - Try partial solution
  - If invalid, backtrack
  - Used for constraint satisfaction

- **Problems**
  - N-Queens
  - Sudoku solver
  - Generate permutations/combinations
  - Subset generation

### 6. Two Pointers Technique
- **Use Cases**
  - Sorted arrays
  - Palindrome checking
  - Pair sum problems
  - Removing duplicates

### 7. Sliding Window
- **Fixed Window**
  - Maximum sum of subarray of size k
  - Average of subarrays

- **Variable Window**
  - Longest substring with k distinct characters
  - Minimum window substring

### 8. Bit Manipulation
- **Common Operations**
  - AND, OR, XOR, NOT
  - Left shift, right shift
  - Check if power of 2
  - Count set bits
  - Find single number in array

## Common Interview Problems

### Problem 1: Two Sum
```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

### Problem 2: Reverse Linked List
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    prev = None
    current = head
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    return prev
```

### Problem 3: Valid Parentheses
```python
def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    return not stack
```

### Problem 4: Merge Two Sorted Lists
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

### Problem 5: Maximum Subarray (Kadane's Algorithm)
```python
def max_subarray(nums):
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum
```

### Problem 6: Binary Search
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
```

### Problem 7: Longest Palindromic Substring
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
```

### Problem 8: Top K Frequent Elements
```python
from collections import Counter
import heapq

def top_k_frequent(nums, k):
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
```

### Problem 9: Product of Array Except Self
```python
def product_except_self(nums):
    n = len(nums)
    result = [1] * n
    
    # Left pass
    for i in range(1, n):
        result[i] = result[i-1] * nums[i-1]
    
    # Right pass
    right = 1
    for i in range(n-1, -1, -1):
        result[i] *= right
        right *= nums[i]
    
    return result
```

### Problem 10: Container With Most Water
```python
def max_area(height):
    left, right = 0, len(height) - 1
    max_water = 0
    while left < right:
        width = right - left
        current_area = min(height[left], height[right]) * width
        max_water = max(max_water, current_area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_water
```

## Complexity Analysis

### Big O Notation
- **Time Complexity**
  - O(1): Constant
  - O(log n): Logarithmic
  - O(n): Linear
  - O(n log n): Linearithmic
  - O(n²): Quadratic
  - O(2ⁿ): Exponential

- **Space Complexity**
  - Auxiliary space
  - Input space
  - Total space complexity

### Amortized Analysis
- Understanding amortized O(1) operations
- Example: list.append() in Python

## Practice Problems by Category

### Arrays & Strings
- Two Sum, Three Sum
- Longest Substring Without Repeating Characters
- Group Anagrams
- Rotate Array
- Merge Intervals

### Linked Lists
- Reverse Linked List
- Merge Two Sorted Lists
- Detect Cycle
- Remove Nth Node From End
- Intersection of Two Linked Lists

### Trees
- Maximum Depth of Binary Tree
- Same Tree
- Invert Binary Tree
- Binary Tree Level Order Traversal
- Validate Binary Search Tree

### Dynamic Programming
- Climbing Stairs
- House Robber
- Coin Change
- Longest Common Subsequence
- Edit Distance

### Graphs
- Number of Islands
- Clone Graph
- Course Schedule
- Word Ladder
