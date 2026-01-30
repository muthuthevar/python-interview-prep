# Stratzi.ai Answers - Approaches + Python Code

This file provides detailed approaches with inline Python solutions for every
problem listed in `stratzi.ai/PRIORITY 0_ MUST PRACTICE (Top 20).md`.

## Helpers

```python
from collections import defaultdict, Counter, deque, OrderedDict
from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class RandomNode:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class GraphNode:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
```

## Priority 0: Must Practice (Top 20)

### Arrays and Hashing

1. Two Sum
   - Approach: One-pass hash map from value to index; for each number, check if its complement is already seen.
   - Edge cases: Duplicate values, same number used twice (avoid reusing same index).
   - Complexity: O(n) time, O(n) space.
   - Code:
```python
def two_sum(nums: List[int], target: int) -> List[int]:
    seen = {}
    for i, x in enumerate(nums):
        need = target - x
        if need in seen:
            return [seen[need], i]
        seen[x] = i
    return []
```

2. Contains Duplicate
   - Approach: Track seen values in a set; return true on first repeat.
   - Edge cases: Large input, negatives, empty list.
   - Complexity: O(n) time, O(n) space.
   - Code:
```python
def contains_duplicate(nums: List[int]) -> bool:
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False
```

3. Valid Anagram
   - Approach: Count characters with fixed array (lowercase) or hash map; compare counts.
   - Edge cases: Different lengths early-exit, unicode if not specified.
   - Complexity: O(n) time, O(1) or O(n) space.
   - Code:
```python
def is_anagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    return Counter(s) == Counter(t)
```

4. Group Anagrams
   - Approach: Use map key as sorted string or 26-count tuple; append strings to buckets.
   - Edge cases: Empty strings, duplicates, long strings (prefer count key).
   - Complexity: O(n k log k) sort-key or O(n k) count-key, O(n) space.
   - Code:
```python
def group_anagrams(strs: List[str]) -> List[List[str]]:
    groups = defaultdict(list)
    for s in strs:
        counts = [0] * 26
        for ch in s:
            counts[ord(ch) - 97] += 1
        groups[tuple(counts)].append(s)
    return list(groups.values())
```

5. Product of Array Except Self
   - Approach: Prefix product pass and suffix product pass; multiply for each index.
   - Edge cases: One or multiple zeros; avoid division.
   - Complexity: O(n) time, O(1) extra space (excluding output).
   - Code:
```python
def product_except_self(nums: List[int]) -> List[int]:
    n = len(nums)
    res = [1] * n
    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        res[i] *= suffix
        suffix *= nums[i]
    return res
```

6. Top K Frequent Elements
   - Approach: Frequency map; then bucket sort by frequency or heap of size k.
   - Edge cases: k equals number of uniques, ties, large n (bucket is linear).
   - Complexity: O(n) time with bucket, O(n log k) with heap.
   - Code:
```python
def top_k_frequent(nums: List[int], k: int) -> List[int]:
    freq = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, f in freq.items():
        buckets[f].append(num)
    res = []
    for f in range(len(buckets) - 1, 0, -1):
        for num in buckets[f]:
            res.append(num)
            if len(res) == k:
                return res
    return res
```

### Two Pointers

7. Valid Palindrome
   - Approach: Two pointers from ends; skip non-alphanumeric; compare lowercased chars.
   - Edge cases: Empty or all non-alnum strings.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def is_palindrome(s: str) -> bool:
    l, r = 0, len(s) - 1
    while l < r:
        while l < r and not s[l].isalnum():
            l += 1
        while l < r and not s[r].isalnum():
            r -= 1
        if s[l].lower() != s[r].lower():
            return False
        l += 1
        r -= 1
    return True
```

8. Two Sum II
   - Approach: Sorted array two pointers; move left/right based on sum.
   - Edge cases: Multiple solutions but return any; 1-indexed output.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def two_sum_sorted(numbers: List[int], target: int) -> List[int]:
    l, r = 0, len(numbers) - 1
    while l < r:
        s = numbers[l] + numbers[r]
        if s == target:
            return [l + 1, r + 1]
        if s < target:
            l += 1
        else:
            r -= 1
    return []
```

9. 3Sum
   - Approach: Sort; for each i, use two pointers on remaining; skip duplicates.
   - Edge cases: All zeros; duplicates across i and pointers.
   - Complexity: O(n^2) time, O(1) extra space (sorting aside).
   - Code:
```python
def three_sum(nums: List[int]) -> List[List[int]]:
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                res.append([nums[i], nums[l], nums[r]])
                l += 1
                r -= 1
                while l < r and nums[l] == nums[l - 1]:
                    l += 1
                while l < r and nums[r] == nums[r + 1]:
                    r -= 1
            elif s < 0:
                l += 1
            else:
                r -= 1
    return res
```

10. Container With Most Water
   - Approach: Two pointers; area = min(height) * width; move shorter side inward.
   - Edge cases: Equal heights; width shrinking.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def max_area(height: List[int]) -> int:
    l, r = 0, len(height) - 1
    best = 0
    while l < r:
        best = max(best, min(height[l], height[r]) * (r - l))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return best
```

### Sliding Window

11. Best Time to Buy and Sell Stock
   - Approach: Track min price so far, update max profit per day.
   - Edge cases: No profit possible -> 0.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def max_profit(prices: List[int]) -> int:
    min_price = float("inf")
    best = 0
    for p in prices:
        min_price = min(min_price, p)
        best = max(best, p - min_price)
    return best
```

12. Longest Substring Without Repeating Characters
   - Approach: Sliding window with last seen index map; move left pointer past repeats.
   - Edge cases: Repeated char far left, all unique.
   - Complexity: O(n) time, O(k) space.
   - Code:
```python
def length_of_longest_substring(s: str) -> int:
    last = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1
        last[ch] = right
        best = max(best, right - left + 1)
    return best
```

### Stack

13. Valid Parentheses
   - Approach: Stack of openers; pop on matching closer; fail on mismatch/empty.
   - Edge cases: Odd length, extra openers at end.
   - Complexity: O(n) time, O(n) space.
   - Code:
```python
def is_valid_parentheses(s: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
        else:
            stack.append(ch)
    return not stack
```

### Binary Search

14. Binary Search
   - Approach: Classic low/high loop, mid compare, narrow bounds.
   - Edge cases: Not found -> -1; overflow safe mid.
   - Complexity: O(log n) time, O(1) space.
   - Code:
```python
def binary_search(nums: List[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target:
            return m
        if nums[m] < target:
            l = m + 1
        else:
            r = m - 1
    return -1
```

15. Search in Rotated Sorted Array
   - Approach: Modified binary search; identify sorted half; check target range.
   - Edge cases: Rotation at ends; target absent.
   - Complexity: O(log n) time, O(1) space.
   - Code:
```python
def search_rotated(nums: List[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target:
            return m
        if nums[l] <= nums[m]:
            if nums[l] <= target < nums[m]:
                r = m - 1
            else:
                l = m + 1
        else:
            if nums[m] < target <= nums[r]:
                l = m + 1
            else:
                r = m - 1
    return -1
```

### Strings

16. Longest Palindromic Substring
   - Approach: Expand around each center (odd and even); track max length.
   - Edge cases: All same char; length 1.
   - Complexity: O(n^2) time, O(1) space.
   - Code:
```python
def longest_palindrome(s: str) -> str:
    if not s:
        return ""
    start = end = 0
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return l + 1, r - 1
    for i in range(len(s)):
        l1, r1 = expand(i, i)
        l2, r2 = expand(i, i + 1)
        if r1 - l1 > end - start:
            start, end = l1, r1
        if r2 - l2 > end - start:
            start, end = l2, r2
    return s[start:end + 1]
```

### Linked List

17. Reverse Linked List
   - Approach: Iterative pointer reversal (prev, curr, next).
   - Edge cases: Empty list, single node.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    prev = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev
```

18. Merge Two Sorted Lists
   - Approach: Dummy head; compare nodes, attach smaller, advance pointer.
   - Edge cases: One list empty; leftover nodes.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def merge_two_lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    cur = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next
```

19. Linked List Cycle
   - Approach: Floyd's tortoise and hare to detect cycle.
   - Edge cases: Short lists; cycle at head.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def has_cycle(head: Optional[ListNode]) -> bool:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

### Trees

20. Maximum Depth of Binary Tree
   - Approach: DFS recursion for max depth; or BFS level count.
   - Edge cases: Empty tree -> 0.
   - Complexity: O(n) time, O(h) recursion or O(n) queue space.
   - Code:
```python
def max_depth(root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

## Priority 1: Important (Next 30)

### Arrays and Hashing

21. Longest Consecutive Sequence
   - Approach: Put all numbers in set; only start counting from sequence starts (num-1 not in set).
   - Edge cases: Duplicates; negative numbers.
   - Complexity: O(n) time, O(n) space.
   - Code:
```python
def longest_consecutive(nums: List[int]) -> int:
    s = set(nums)
    best = 0
    for x in s:
        if x - 1 not in s:
            cur = x
            length = 1
            while cur + 1 in s:
                cur += 1
                length += 1
            best = max(best, length)
    return best
```

22. Valid Sudoku
   - Approach: Track seen digits per row, column, and 3x3 box using sets.
   - Edge cases: Ignore dots; invalid box indexing.
   - Complexity: O(1) time and space (fixed 9x9).
   - Code:
```python
def is_valid_sudoku(board: List[List[str]]) -> bool:
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if v == ".":
                continue
            b = (r // 3) * 3 + (c // 3)
            if v in rows[r] or v in cols[c] or v in boxes[b]:
                return False
            rows[r].add(v)
            cols[c].add(v)
            boxes[b].add(v)
    return True
```

### Two Pointers

23. Remove Duplicates from Sorted Array
   - Approach: Slow pointer for next unique; fast pointer scans.
   - Edge cases: All unique or all same.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def remove_duplicates(nums: List[int]) -> int:
    if not nums:
        return 0
    k = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            nums[k] = nums[i]
            k += 1
    return k
```

24. Trapping Rain Water
   - Approach: Two pointers with left/right max; move lower side and add trapped water.
   - Edge cases: Monotonic heights, small arrays.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def trap(height: List[int]) -> int:
    l, r = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    while l < r:
        if height[l] < height[r]:
            left_max = max(left_max, height[l])
            water += left_max - height[l]
            l += 1
        else:
            right_max = max(right_max, height[r])
            water += right_max - height[r]
            r -= 1
    return water
```

### Sliding Window

25. Longest Repeating Character Replacement
   - Approach: Window with char counts; keep max freq; shrink if window_len - max_freq > k.
   - Edge cases: Update max freq lazily; k = 0.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def character_replacement(s: str, k: int) -> int:
    counts = defaultdict(int)
    left = 0
    max_freq = 0
    best = 0
    for right, ch in enumerate(s):
        counts[ch] += 1
        max_freq = max(max_freq, counts[ch])
        while (right - left + 1) - max_freq > k:
            counts[s[left]] -= 1
            left += 1
        best = max(best, right - left + 1)
    return best
```

26. Permutation in String
   - Approach: Fixed-size window counts; compare to target counts.
   - Edge cases: Window length > s2 length.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def check_inclusion(s1: str, s2: str) -> bool:
    if len(s1) > len(s2):
        return False
    need = [0] * 26
    for ch in s1:
        need[ord(ch) - 97] += 1
    window = [0] * 26
    for i, ch in enumerate(s2):
        window[ord(ch) - 97] += 1
        if i >= len(s1):
            window[ord(s2[i - len(s1)]) - 97] -= 1
        if window == need:
            return True
    return False
```

27. Minimum Window Substring
   - Approach: Sliding window with need counts; expand to satisfy, shrink to minimize.
   - Edge cases: No valid window; repeated chars.
   - Complexity: O(n) time, O(1) or O(k) space.
   - Code:
```python
def min_window(s: str, t: str) -> str:
    if not t:
        return ""
    need = Counter(t)
    have = 0
    required = len(need)
    left = 0
    best = (float("inf"), 0, 0)
    window = defaultdict(int)
    for right, ch in enumerate(s):
        window[ch] += 1
        if ch in need and window[ch] == need[ch]:
            have += 1
        while have == required:
            if right - left + 1 < best[0]:
                best = (right - left + 1, left, right)
            window[s[left]] -= 1
            if s[left] in need and window[s[left]] < need[s[left]]:
                have -= 1
            left += 1
    return "" if best[0] == float("inf") else s[best[1]:best[2] + 1]
```

### Stack

28. Min Stack
   - Approach: Stack of values plus stack of current mins (or store pairs).
   - Edge cases: Duplicate minimum values.
   - Complexity: O(1) per op, O(n) space.
   - Code:
```python
class MinStack:
    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        cur_min = val if not self.stack else min(val, self.stack[-1][1])
        self.stack.append((val, cur_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

29. Evaluate Reverse Polish Notation
   - Approach: Stack; on operator, pop two, apply, push result.
   - Edge cases: Integer division truncation rules.
   - Complexity: O(n) time, O(n) space.
   - Code:
```python
def eval_rpn(tokens: List[str]) -> int:
    stack = []
    for tok in tokens:
        if tok in {"+", "-", "*", "/"}:
            b = stack.pop()
            a = stack.pop()
            if tok == "+":
                stack.append(a + b)
            elif tok == "-":
                stack.append(a - b)
            elif tok == "*":
                stack.append(a * b)
            else:
                stack.append(int(a / b))
        else:
            stack.append(int(tok))
    return stack[-1]
```

30. Daily Temperatures
   - Approach: Monotonic decreasing stack of indices; resolve when warmer day found.
   - Edge cases: No warmer day -> 0.
   - Complexity: O(n) time, O(n) space.
   - Code:
```python
def daily_temperatures(temperatures: List[int]) -> List[int]:
    res = [0] * len(temperatures)
    stack = []
    for i, t in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < t:
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
```

### Binary Search

31. Find Minimum in Rotated Sorted Array
   - Approach: Binary search; compare mid to right to find unsorted half.
   - Edge cases: Not rotated; small arrays.
   - Complexity: O(log n) time, O(1) space.
   - Code:
```python
def find_min(nums: List[int]) -> int:
    l, r = 0, len(nums) - 1
    while l < r:
        m = (l + r) // 2
        if nums[m] > nums[r]:
            l = m + 1
        else:
            r = m
    return nums[l]
```

32. Search a 2D Matrix
   - Approach: Treat matrix as sorted array; binary search by index mapping.
   - Edge cases: Empty matrix.
   - Complexity: O(log(mn)) time, O(1) space.
   - Code:
```python
def search_matrix(matrix: List[List[int]], target: int) -> bool:
    if not matrix or not matrix[0]:
        return False
    m, n = len(matrix), len(matrix[0])
    l, r = 0, m * n - 1
    while l <= r:
        mid = (l + r) // 2
        val = matrix[mid // n][mid % n]
        if val == target:
            return True
        if val < target:
            l = mid + 1
        else:
            r = mid - 1
    return False
```

33. Koko Eating Bananas
   - Approach: Binary search on speed k; check if total hours <= h.
   - Edge cases: h equals piles count; large piles.
   - Complexity: O(n log M) time, O(1) space.
   - Code:
```python
def min_eating_speed(piles: List[int], h: int) -> int:
    l, r = 1, max(piles)
    def can(k):
        return sum((p + k - 1) // k for p in piles) <= h
    while l < r:
        m = (l + r) // 2
        if can(m):
            r = m
        else:
            l = m + 1
    return l
```

### Linked List

34. Reorder List
   - Approach: Find middle, reverse second half, merge alternating nodes.
   - Edge cases: Odd/even length, two nodes.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def reorder_list(head: Optional[ListNode]) -> None:
    if not head or not head.next:
        return
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    prev = None
    cur = slow.next
    slow.next = None
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first, second = tmp1, tmp2
```

35. Remove Nth Node From End
   - Approach: Two pointers with gap n; move together to find node before target.
   - Edge cases: Remove head; n equals length.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n):
        fast = fast.next
    while fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next
```

36. Copy List with Random Pointer
   - Approach: Hash map old->new, or interleave copied nodes for O(1) extra.
   - Edge cases: Random is null; self-random.
   - Complexity: O(n) time, O(n) or O(1) space.
   - Code:
```python
def copy_random_list(head: Optional[RandomNode]) -> Optional[RandomNode]:
    if not head:
        return None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = RandomNode(cur.val, nxt)
        cur = nxt
    cur = head
    while cur:
        if cur.random:
            cur.next.random = cur.random.next
        cur = cur.next.next
    cur = head
    new_head = head.next
    while cur:
        copy = cur.next
        cur.next = copy.next
        copy.next = copy.next.next if copy.next else None
        cur = cur.next
    return new_head
```

37. Add Two Numbers
   - Approach: Traverse both lists with carry; build result list.
   - Edge cases: Different lengths; carry at end.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def add_two_numbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    cur = dummy
    carry = 0
    while l1 or l2 or carry:
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0
        total = v1 + v2 + carry
        carry = total // 10
        cur.next = ListNode(total % 10)
        cur = cur.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next
```

38. LRU Cache
   - Approach: Hash map + doubly linked list for O(1) get/put; evict tail.
   - Edge cases: Update existing key; capacity 1.
   - Complexity: O(1) per op, O(n) space.
   - Code:
```python
class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
```

### Trees

39. Invert Binary Tree
   - Approach: DFS swap left/right recursively; or BFS.
   - Edge cases: Empty tree.
   - Complexity: O(n) time, O(h) space.
   - Code:
```python
def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
```

40. Diameter of Binary Tree
   - Approach: Postorder DFS; track max left_depth + right_depth; return height.
   - Edge cases: Single node.
   - Complexity: O(n) time, O(h) space.
   - Code:
```python
def diameter_of_binary_tree(root: Optional[TreeNode]) -> int:
    diameter = 0
    def depth(node):
        nonlocal diameter
        if not node:
            return 0
        left = depth(node.left)
        right = depth(node.right)
        diameter = max(diameter, left + right)
        return 1 + max(left, right)
    depth(root)
    return diameter
```

41. Balanced Binary Tree
   - Approach: DFS returns height; if subtree imbalance > 1, bubble up failure.
   - Edge cases: Unbalanced near leaves.
   - Complexity: O(n) time, O(h) space.
   - Code:
```python
def is_balanced(root: Optional[TreeNode]) -> bool:
    def height(node):
        if not node:
            return 0
        left = height(node.left)
        if left == -1:
            return -1
        right = height(node.right)
        if right == -1 or abs(left - right) > 1:
            return -1
        return 1 + max(left, right)
    return height(root) != -1
```

42. Same Tree
   - Approach: DFS compare structure and values.
   - Edge cases: One null, one not.
   - Complexity: O(n) time, O(h) space.
   - Code:
```python
def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
```

43. Subtree of Another Tree
   - Approach: Traverse root; when values match, run same-tree check.
   - Edge cases: Empty subtree -> true.
   - Complexity: O(n*m) worst, O(n) average with pruning.
   - Code:
```python
def is_subtree(root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    if not subRoot:
        return True
    if not root:
        return False
    if is_same_tree(root, subRoot):
        return True
    return is_subtree(root.left, subRoot) or is_subtree(root.right, subRoot)
```

44. Binary Tree Level Order Traversal
   - Approach: BFS queue; process by level size.
   - Edge cases: Empty tree.
   - Complexity: O(n) time, O(n) space.
   - Code:
```python
def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    res = []
    q = deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level)
    return res
```

45. Binary Tree Right Side View
   - Approach: BFS by level, take last node; or DFS with depth priority.
   - Edge cases: Single node.
   - Complexity: O(n) time, O(n) space.
   - Code:
```python
def right_side_view(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    res = []
    q = deque([root])
    while q:
        rightmost = None
        for _ in range(len(q)):
            node = q.popleft()
            rightmost = node.val
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(rightmost)
    return res
```

46. Validate Binary Search Tree
   - Approach: DFS with valid value range (min, max).
   - Edge cases: Duplicates not allowed; large values.
   - Complexity: O(n) time, O(h) space.
   - Code:
```python
def is_valid_bst(root: Optional[TreeNode]) -> bool:
    def dfs(node, low, high):
        if not node:
            return True
        if not (low < node.val < high):
            return False
        return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)
    return dfs(root, float("-inf"), float("inf"))
```

47. Kth Smallest in BST
   - Approach: Inorder traversal yields sorted order; stop at k.
   - Edge cases: k at extremes.
   - Complexity: O(h + k) time, O(h) space.
   - Code:
```python
def kth_smallest(root: Optional[TreeNode], k: int) -> int:
    stack = []
    cur = root
    while True:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        k -= 1
        if k == 0:
            return cur.val
        cur = cur.right
```

48. Lowest Common Ancestor of BST
   - Approach: Traverse from root; go left/right based on p and q values.
   - Edge cases: One node is ancestor of other.
   - Complexity: O(h) time, O(1) space.
   - Code:
```python
def lowest_common_ancestor_bst(root: Optional[TreeNode], p: TreeNode, q: TreeNode) -> Optional[TreeNode]:
    cur = root
    while cur:
        if p.val < cur.val and q.val < cur.val:
            cur = cur.left
        elif p.val > cur.val and q.val > cur.val:
            cur = cur.right
        else:
            return cur
    return None
```

### Graphs

49. Number of Islands
   - Approach: DFS/BFS from each land cell, mark visited.
   - Edge cases: All water; large grids.
   - Complexity: O(mn) time, O(mn) space.
   - Code:
```python
def num_islands(grid: List[List[str]]) -> int:
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    def dfs(r, c):
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != "1":
            return
        grid[r][c] = "0"
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    count = 0
    for r in range(m):
        for c in range(n):
            if grid[r][c] == "1":
                count += 1
                dfs(r, c)
    return count
```

50. Clone Graph
   - Approach: DFS/BFS with map from original to clone nodes.
   - Edge cases: Single node, cycles.
   - Complexity: O(V+E) time, O(V) space.
   - Code:
```python
def clone_graph(node: Optional[GraphNode]) -> Optional[GraphNode]:
    if not node:
        return None
    clones = {node: GraphNode(node.val)}
    q = deque([node])
    while q:
        cur = q.popleft()
        for nei in cur.neighbors:
            if nei not in clones:
                clones[nei] = GraphNode(nei.val)
                q.append(nei)
            clones[cur].neighbors.append(clones[nei])
    return clones[node]
```

## Priority 2: Good To Know (Next 27)

### Strings

51. Palindromic Substrings
   - Approach: Expand around center for each index (odd/even).
   - Edge cases: All same char.
   - Complexity: O(n^2) time, O(1) space.
   - Code:
```python
def count_substrings(s: str) -> int:
    def expand(l, r):
        count = 0
        while l >= 0 and r < len(s) and s[l] == s[r]:
            count += 1
            l -= 1
            r += 1
        return count
    total = 0
    for i in range(len(s)):
        total += expand(i, i)
        total += expand(i, i + 1)
    return total
```

52. Encode and Decode Strings
   - Approach: Length-prefix each string; decode by reading length then content.
   - Edge cases: Empty strings, separators inside strings.
   - Complexity: O(n) time, O(1) extra space.
   - Code:
```python
def encode(strs: List[str]) -> str:
    return "".join(f"{len(s)}#{s}" for s in strs)


def decode(s: str) -> List[str]:
    res = []
    i = 0
    while i < len(s):
        j = i
        while s[j] != "#":
            j += 1
        length = int(s[i:j])
        res.append(s[j + 1:j + 1 + length])
        i = j + 1 + length
    return res
```

### Graphs

53. Max Area of Island
   - Approach: DFS/BFS count area for each island, track max.
   - Edge cases: All water.
   - Complexity: O(mn) time, O(mn) space.
   - Code:
```python
def max_area_of_island(grid: List[List[int]]) -> int:
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    def dfs(r, c):
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != 1:
            return 0
        grid[r][c] = 0
        return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)
    best = 0
    for r in range(m):
        for c in range(n):
            if grid[r][c] == 1:
                best = max(best, dfs(r, c))
    return best
```

54. Pacific Atlantic Water Flow
   - Approach: Reverse flow; BFS/DFS from both oceans, intersect reachable cells.
   - Edge cases: Equal heights; boundaries.
   - Complexity: O(mn) time, O(mn) space.
   - Code:
```python
def pacific_atlantic(heights: List[List[int]]) -> List[List[int]]:
    if not heights:
        return []
    m, n = len(heights), len(heights[0])
    pac = [[False] * n for _ in range(m)]
    atl = [[False] * n for _ in range(m)]
    def dfs(r, c, ocean):
        ocean[r][c] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and not ocean[nr][nc] and heights[nr][nc] >= heights[r][c]:
                dfs(nr, nc, ocean)
    for r in range(m):
        dfs(r, 0, pac)
        dfs(r, n - 1, atl)
    for c in range(n):
        dfs(0, c, pac)
        dfs(m - 1, c, atl)
    res = []
    for r in range(m):
        for c in range(n):
            if pac[r][c] and atl[r][c]:
                res.append([r, c])
    return res
```

55. Course Schedule
   - Approach: Detect cycle in directed graph via DFS states or Kahn's topo sort.
   - Edge cases: Disconnected graph.
   - Complexity: O(V+E) time, O(V+E) space.
   - Code:
```python
def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
    graph = defaultdict(list)
    indeg = [0] * num_courses
    for a, b in prerequisites:
        graph[b].append(a)
        indeg[a] += 1
    q = deque([i for i in range(num_courses) if indeg[i] == 0])
    taken = 0
    while q:
        c = q.popleft()
        taken += 1
        for nei in graph[c]:
            indeg[nei] -= 1
            if indeg[nei] == 0:
                q.append(nei)
    return taken == num_courses
```

56. Course Schedule II
   - Approach: Topological sort; if cycle, return empty.
   - Edge cases: Multiple valid orders.
   - Complexity: O(V+E) time, O(V+E) space.
   - Code:
```python
def find_order(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    graph = defaultdict(list)
    indeg = [0] * num_courses
    for a, b in prerequisites:
        graph[b].append(a)
        indeg[a] += 1
    q = deque([i for i in range(num_courses) if indeg[i] == 0])
    order = []
    while q:
        c = q.popleft()
        order.append(c)
        for nei in graph[c]:
            indeg[nei] -= 1
            if indeg[nei] == 0:
                q.append(nei)
    return order if len(order) == num_courses else []
```

### Trees

57. Construct Binary Tree from Preorder and Inorder
   - Approach: Root from preorder; split inorder by root index; recurse.
   - Edge cases: Single node; use map for inorder indices.
   - Complexity: O(n) time, O(n) space.
   - Code:
```python
def build_tree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    index = {v: i for i, v in enumerate(inorder)}
    def helper(pl, pr, il, ir):
        if pl > pr:
            return None
        root_val = preorder[pl]
        root = TreeNode(root_val)
        mid = index[root_val]
        left_size = mid - il
        root.left = helper(pl + 1, pl + left_size, il, mid - 1)
        root.right = helper(pl + left_size + 1, pr, mid + 1, ir)
        return root
    return helper(0, len(preorder) - 1, 0, len(inorder) - 1)
```

### 1-D Dynamic Programming

58. Climbing Stairs
   - Approach: Fibonacci DP; ways[i] = ways[i-1] + ways[i-2].
   - Edge cases: n <= 2.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def climb_stairs(n: int) -> int:
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b
```

59. Min Cost Climbing Stairs
   - Approach: DP for min cost to reach each step; start at 0 or 1.
   - Edge cases: Small n.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def min_cost_climbing_stairs(cost: List[int]) -> int:
    a, b = 0, 0
    for c in cost:
        a, b = b, min(a, b) + c
    return min(a, b)
```

60. House Robber
   - Approach: DP with include/exclude; max at each house.
   - Edge cases: One house.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def rob(nums: List[int]) -> int:
    prev = curr = 0
    for n in nums:
        prev, curr = curr, max(curr, prev + n)
    return curr
```

61. House Robber II
   - Approach: Run House Robber twice excluding first or last; take max.
   - Edge cases: n == 1.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def rob_ii(nums: List[int]) -> int:
    if len(nums) == 1:
        return nums[0]
    def rob_line(arr):
        prev = curr = 0
        for n in arr:
            prev, curr = curr, max(curr, prev + n)
        return curr
    return max(rob_line(nums[:-1]), rob_line(nums[1:]))
```

62. Decode Ways
   - Approach: DP on string; dp[i] from valid 1- or 2-digit decodes.
   - Edge cases: Leading zeros; "0" invalid alone.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def num_decodings(s: str) -> int:
    if not s or s[0] == "0":
        return 0
    prev2, prev1 = 1, 1
    for i in range(1, len(s)):
        cur = 0
        if s[i] != "0":
            cur += prev1
        two = int(s[i - 1:i + 1])
        if 10 <= two <= 26:
            cur += prev2
        prev2, prev1 = prev1, cur
    return prev1
```

63. Coin Change
   - Approach: DP for min coins; dp[0]=0, update dp[x] from dp[x-coin].
   - Edge cases: Unreachable amount -> -1.
   - Complexity: O(n*amount) time, O(amount) space.
   - Code:
```python
def coin_change(coins: List[int], amount: int) -> int:
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for c in coins:
        for x in range(c, amount + 1):
            dp[x] = min(dp[x], dp[x - c] + 1)
    return -1 if dp[amount] == float("inf") else dp[amount]
```

64. Maximum Product Subarray
   - Approach: Track current max and min (negative flips).
   - Edge cases: Zeros split; all negatives.
   - Complexity: O(n) time, O(1) space.
   - Code:
```python
def max_product(nums: List[int]) -> int:
    cur_max = cur_min = ans = nums[0]
    for n in nums[1:]:
        if n < 0:
            cur_max, cur_min = cur_min, cur_max
        cur_max = max(n, cur_max * n)
        cur_min = min(n, cur_min * n)
        ans = max(ans, cur_max)
    return ans
```

65. Word Break
   - Approach: DP boolean; dp[i] true if any word ends at i.
   - Edge cases: Large dictionary; reuse words.
   - Complexity: O(n * k) time, O(n) space.
   - Code:
```python
def word_break(s: str, word_dict: List[str]) -> bool:
    word_set = set(word_dict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[-1]
```

66. Longest Increasing Subsequence
   - Approach: Patience sorting with tails array; binary search insert.
   - Edge cases: Duplicates not increasing.
   - Complexity: O(n log n) time, O(n) space.
   - Code:
```python
def length_of_lis(nums: List[int]) -> int:
    tails = []
    for n in nums:
        l, r = 0, len(tails)
        while l < r:
            m = (l + r) // 2
            if tails[m] < n:
                l = m + 1
            else:
                r = m
        if l == len(tails):
            tails.append(n)
        else:
            tails[l] = n
    return len(tails)
```

### Intervals

67. Insert Interval
   - Approach: Add all intervals before, merge overlaps, append rest.
   - Edge cases: New interval before/after all.
   - Complexity: O(n) time, O(n) space.
   - Code:
```python
def insert(intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    res = []
    i = 0
    while i < len(intervals) and intervals[i][1] < new_interval[0]:
        res.append(intervals[i])
        i += 1
    while i < len(intervals) and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    res.append(new_interval)
    while i < len(intervals):
        res.append(intervals[i])
        i += 1
    return res
```

68. Merge Intervals
   - Approach: Sort by start; merge when current overlaps last.
   - Edge cases: Touching endpoints.
   - Complexity: O(n log n) time, O(n) space.
   - Code:
```python
def merge(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort()
    res = []
    for s, e in intervals:
        if not res or res[-1][1] < s:
            res.append([s, e])
        else:
            res[-1][1] = max(res[-1][1], e)
    return res
```

69. Non-overlapping Intervals
   - Approach: Sort by end time; count removals when overlap (greedy).
   - Edge cases: Equal ends.
   - Complexity: O(n log n) time, O(1) extra space.
   - Code:
```python
def erase_overlap_intervals(intervals: List[List[int]]) -> int:
    intervals.sort(key=lambda x: x[1])
    end = float("-inf")
    count = 0
    for s, e in intervals:
        if s >= end:
            end = e
        else:
            count += 1
    return count
```

70. Meeting Rooms
   - Approach: Sort intervals by start; check for any overlap.
   - Edge cases: Single meeting.
   - Complexity: O(n log n) time, O(1) space.
   - Code:
```python
def can_attend_meetings(intervals: List[List[int]]) -> bool:
    intervals.sort()
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i - 1][1]:
            return False
    return True
```

71. Meeting Rooms II
   - Approach: Min-heap of end times; allocate rooms when overlap.
   - Edge cases: Meetings end exactly when another starts.
   - Complexity: O(n log n) time, O(n) space.
   - Code:
```python
import heapq


def min_meeting_rooms(intervals: List[List[int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    heap = [intervals[0][1]]
    for s, e in intervals[1:]:
        if s >= heap[0]:
            heapq.heapreplace(heap, e)
        else:
            heapq.heappush(heap, e)
    return len(heap)
```

### Backtracking

72. Subsets
   - Approach: Backtrack include/exclude each element; or iterative build.
   - Edge cases: Empty set.
   - Complexity: O(n * 2^n) time, O(n) space.
   - Code:
```python
def subsets(nums: List[int]) -> List[List[int]]:
    res = [[]]
    for n in nums:
        res += [cur + [n] for cur in res]
    return res
```

73. Combination Sum
   - Approach: Backtrack with non-decreasing choices; allow reuse of same number.
   - Edge cases: Avoid duplicates by index progression.
   - Complexity: Exponential time, O(target) space.
   - Code:
```python
def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    res = []
    candidates.sort()
    def backtrack(start, path, total):
        if total == target:
            res.append(path[:])
            return
        if total > target:
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, path, total + candidates[i])
            path.pop()
    backtrack(0, [], 0)
    return res
```

74. Permutations
   - Approach: Backtrack swapping or used set; build permutations.
   - Edge cases: Duplicates (if any) need handling.
   - Complexity: O(n * n!) time, O(n) space.
   - Code:
```python
def permute(nums: List[int]) -> List[List[int]]:
    res = []
    def backtrack(start):
        if start == len(nums):
            res.append(nums[:])
            return
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
    backtrack(0)
    return res
```

75. Letter Combinations of a Phone Number
   - Approach: Backtrack over digits, append each mapped letter.
   - Edge cases: Empty input -> empty list.
   - Complexity: O(4^n) time, O(n) space.
   - Code:
```python
def letter_combinations(digits: str) -> List[str]:
    if not digits:
        return []
    phone = {
        "2": "abc", "3": "def", "4": "ghi", "5": "jkl",
        "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"
    }
    res = []
    def backtrack(i, path):
        if i == len(digits):
            res.append("".join(path))
            return
        for ch in phone[digits[i]]:
            path.append(ch)
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return res
```

76. Word Search
   - Approach: DFS from each cell; mark visited; prune on mismatch.
   - Edge cases: Reuse of cell not allowed; early exit.
   - Complexity: O(mn * 4^L) time, O(L) space.
   - Code:
```python
def exist(board: List[List[str]], word: str) -> bool:
    m, n = len(board), len(board[0])
    def dfs(r, c, i):
        if i == len(word):
            return True
        if r < 0 or r >= m or c < 0 or c >= n or board[r][c] != word[i]:
            return False
        tmp = board[r][c]
        board[r][c] = "#"
        found = (dfs(r + 1, c, i + 1) or dfs(r - 1, c, i + 1) or
                 dfs(r, c + 1, i + 1) or dfs(r, c - 1, i + 1))
        board[r][c] = tmp
        return found
    for r in range(m):
        for c in range(n):
            if dfs(r, c, 0):
                return True
    return False
```

### Stack

77. Car Fleet
   - Approach: Sort cars by position descending; compute time to target; use stack to count fleets.
   - Edge cases: Same position; faster car behind slower one.
   - Complexity: O(n log n) time, O(n) space.
   - Code:
```python
def car_fleet(target: int, position: List[int], speed: List[int]) -> int:
    cars = sorted(zip(position, speed))
    stack = []
    for p, s in reversed(cars):
        time = (target - p) / s
        if not stack or time > stack[-1]:
            stack.append(time)
    return len(stack)
```

## Quick Start (< 3 hours before interview)

These are covered above; use the referenced entries for details:
1. Two Sum (see #1)
2. Valid Parentheses (see #13)
3. Best Time to Buy and Sell Stock (see #11)
4. Valid Palindrome (see #7)
5. Longest Substring Without Repeating Characters (see #12)
6. Reverse Linked List (see #17)
7. Maximum Depth of Binary Tree (see #20)
8. Valid Anagram (see #3)
9. Contains Duplicate (see #2)
10. Merge Two Sorted Lists (see #18)
