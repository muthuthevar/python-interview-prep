"""
CRITICAL DSA PROBLEMS - MUST KNOW FOR INTERVIEW
These are the problems you absolutely cannot fail. Master these patterns.
"""

# ============================================================================
# ARRAYS & STRINGS - ABSOLUTE ESSENTIALS
# ============================================================================

def two_sum(nums, target):
    """
    PROBLEM: Find two numbers that add up to target
    PATTERN: Hash map for O(1) lookup
    TIME: O(n), SPACE: O(n)
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


def three_sum(nums):
    """
    PROBLEM: Find all unique triplets that sum to zero
    PATTERN: Sort + Two pointers
    TIME: O(n²), SPACE: O(1)
    """
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        # Skip duplicates
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    
    return result


def longest_substring_without_repeating(s):
    """
    PROBLEM: Length of longest substring without repeating characters
    PATTERN: Sliding window with hash map
    TIME: O(n), SPACE: O(min(n, m)) where m is charset size
    """
    char_map = {}
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        if char in char_map and char_map[char] >= start:
            start = char_map[char] + 1
        char_map[char] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length


def group_anagrams(strs):
    """
    PROBLEM: Group strings that are anagrams
    PATTERN: Hash map with sorted string as key
    TIME: O(n * k log k), SPACE: O(n * k)
    """
    from collections import defaultdict
    
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())


def product_except_self(nums):
    """
    PROBLEM: Product of array except self (no division)
    PATTERN: Prefix and suffix products
    TIME: O(n), SPACE: O(1) excluding output
    """
    n = len(nums)
    result = [1] * n
    
    # Left pass - prefix products
    for i in range(1, n):
        result[i] = result[i - 1] * nums[i - 1]
    
    # Right pass - suffix products
    right = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right
        right *= nums[i]
    
    return result


def container_with_most_water(height):
    """
    PROBLEM: Container with most water
    PATTERN: Two pointers from both ends
    TIME: O(n), SPACE: O(1)
    """
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
        width = right - left
        area = min(height[left], height[right]) * width
        max_water = max(max_water, area)
        
        # Move pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water


def rotate_array(nums, k):
    """
    PROBLEM: Rotate array to right by k steps (in-place)
    PATTERN: Reverse entire array, then reverse parts
    TIME: O(n), SPACE: O(1)
    """
    n = len(nums)
    k = k % n
    
    # Reverse entire array
    nums.reverse()
    # Reverse first k elements
    nums[:k] = reversed(nums[:k])
    # Reverse remaining elements
    nums[k:] = reversed(nums[k:])


def merge_intervals(intervals):
    """
    PROBLEM: Merge overlapping intervals
    PATTERN: Sort by start, then merge
    TIME: O(n log n), SPACE: O(n)
    """
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            # Overlapping, merge
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            merged.append(current)
    
    return merged


# ============================================================================
# LINKED LISTS - MUST KNOW
# ============================================================================

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_linked_list(head):
    """
    PROBLEM: Reverse a linked list
    PATTERN: Iterative with three pointers
    TIME: O(n), SPACE: O(1)
    """
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    return prev


def merge_two_sorted_lists(l1, l2):
    """
    PROBLEM: Merge two sorted linked lists
    PATTERN: Two pointers with dummy node
    TIME: O(n + m), SPACE: O(1)
    """
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


def has_cycle(head):
    """
    PROBLEM: Detect cycle in linked list
    PATTERN: Floyd's cycle detection (tortoise and hare)
    TIME: O(n), SPACE: O(1)
    """
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    
    return False


def remove_nth_from_end(head, n):
    """
    PROBLEM: Remove nth node from end
    PATTERN: Two pointers with n gap
    TIME: O(n), SPACE: O(1)
    """
    dummy = ListNode(0)
    dummy.next = head
    first = second = dummy
    
    # Move first pointer n+1 steps ahead
    for _ in range(n + 1):
        first = first.next
    
    # Move both until first reaches end
    while first:
        first = first.next
        second = second.next
    
    # Remove node
    second.next = second.next.next
    return dummy.next


def add_two_numbers(l1, l2):
    """
    PROBLEM: Add two numbers represented as linked lists
    PATTERN: Simulate addition with carry
    TIME: O(max(n, m)), SPACE: O(max(n, m))
    """
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        current.next = ListNode(total % 10)
        current = current.next
        
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    
    return dummy.next


# ============================================================================
# STACKS & QUEUES - CRITICAL
# ============================================================================

def valid_parentheses(s):
    """
    PROBLEM: Valid parentheses
    PATTERN: Stack for matching
    TIME: O(n), SPACE: O(n)
    """
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
    """
    PROBLEM: Days until warmer temperature
    PATTERN: Monotonic decreasing stack
    TIME: O(n), SPACE: O(n)
    """
    stack = []
    result = [0] * len(temperatures)
    
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        stack.append(i)
    
    return result


def largest_rectangle_in_histogram(heights):
    """
    PROBLEM: Largest rectangle in histogram
    PATTERN: Monotonic increasing stack
    TIME: O(n), SPACE: O(n)
    """
    stack = []
    max_area = 0
    
    for i, height in enumerate(heights):
        while stack and heights[stack[-1]] > height:
            h = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * width)
        stack.append(i)
    
    # Process remaining bars
    while stack:
        h = heights[stack.pop()]
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        max_area = max(max_area, h * width)
    
    return max_area


# ============================================================================
# DYNAMIC PROGRAMMING - ESSENTIAL PATTERNS
# ============================================================================

def climb_stairs(n):
    """
    PROBLEM: Ways to climb n stairs (1 or 2 steps)
    PATTERN: Fibonacci sequence
    TIME: O(n), SPACE: O(1)
    """
    if n <= 2:
        return n
    
    first, second = 1, 2
    for i in range(3, n + 1):
        first, second = second, first + second
    
    return second


def max_subarray(nums):
    """
    PROBLEM: Maximum sum subarray (Kadane's algorithm)
    PATTERN: Keep track of maximum ending at each position
    TIME: O(n), SPACE: O(1)
    """
    max_sum = current_sum = nums[0]
    
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum


def house_robber(nums):
    """
    PROBLEM: Maximum money without robbing adjacent houses
    PATTERN: DP - choose or skip
    TIME: O(n), SPACE: O(1)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2, prev1 = nums[0], max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, current
    
    return prev1


def coin_change(coins, amount):
    """
    PROBLEM: Minimum coins to make amount
    PATTERN: DP - bottom up
    TIME: O(amount * len(coins)), SPACE: O(amount)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def longest_common_subsequence(text1, text2):
    """
    PROBLEM: Length of longest common subsequence
    PATTERN: 2D DP
    TIME: O(m * n), SPACE: O(m * n)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def edit_distance(word1, word2):
    """
    PROBLEM: Minimum operations to convert word1 to word2
    PATTERN: 2D DP
    TIME: O(m * n), SPACE: O(m * n)
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )
    
    return dp[m][n]


# ============================================================================
# BINARY SEARCH - MUST KNOW
# ============================================================================

def binary_search(nums, target):
    """
    PROBLEM: Search in sorted array
    PATTERN: Binary search
    TIME: O(log n), SPACE: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def search_rotated_array(nums, target):
    """
    PROBLEM: Search in rotated sorted array
    PATTERN: Modified binary search
    TIME: O(log n), SPACE: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1


def find_first_last_position(nums, target):
    """
    PROBLEM: Find first and last position of target in sorted array
    PATTERN: Binary search for boundaries
    TIME: O(log n), SPACE: O(1)
    """
    def find_boundary(is_first):
        left, right = 0, len(nums) - 1
        idx = -1
        
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                idx = mid
                if is_first:
                    right = mid - 1
                else:
                    left = mid + 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return idx
    
    first = find_boundary(True)
    if first == -1:
        return [-1, -1]
    last = find_boundary(False)
    return [first, last]


# ============================================================================
# TREES - ESSENTIAL
# ============================================================================

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root):
    """
    PROBLEM: Maximum depth of binary tree
    PATTERN: DFS recursion
    TIME: O(n), SPACE: O(h) where h is height
    """
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


def same_tree(p, q):
    """
    PROBLEM: Check if two trees are same
    PATTERN: DFS recursion
    TIME: O(n), SPACE: O(h)
    """
    if not p and not q:
        return True
    if not p or not q:
        return False
    return (p.val == q.val and 
            same_tree(p.left, q.left) and 
            same_tree(p.right, q.right))


def invert_tree(root):
    """
    PROBLEM: Invert binary tree
    PATTERN: DFS recursion
    TIME: O(n), SPACE: O(h)
    """
    if not root:
        return None
    
    root.left, root.right = root.right, root.left
    invert_tree(root.left)
    invert_tree(root.right)
    return root


def level_order_traversal(root):
    """
    PROBLEM: Level order traversal (BFS)
    PATTERN: Queue-based BFS
    TIME: O(n), SPACE: O(n)
    """
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


def validate_bst(root, min_val=float('-inf'), max_val=float('inf')):
    """
    PROBLEM: Validate binary search tree
    PATTERN: DFS with bounds
    TIME: O(n), SPACE: O(h)
    """
    if not root:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return (validate_bst(root.left, min_val, root.val) and
            validate_bst(root.right, root.val, max_val))


def lowest_common_ancestor(root, p, q):
    """
    PROBLEM: Lowest common ancestor in BST
    PATTERN: Use BST property
    TIME: O(h), SPACE: O(1)
    """
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
    return None


# ============================================================================
# GRAPHS - CRITICAL
# ============================================================================

def num_islands(grid):
    """
    PROBLEM: Number of islands
    PATTERN: DFS on 2D grid
    TIME: O(m * n), SPACE: O(m * n)
    """
    if not grid:
        return 0
    
    def dfs(i, j):
        if (i < 0 or i >= len(grid) or 
            j < 0 or j >= len(grid[0]) or 
            grid[i][j] != '1'):
            return
        
        grid[i][j] = '0'
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
    
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    
    return count


def clone_graph(node):
    """
    PROBLEM: Clone undirected graph
    PATTERN: BFS with hash map
    TIME: O(n), SPACE: O(n)
    """
    if not node:
        return None
    
    from collections import deque
    
    clone_map = {}
    queue = deque([node])
    clone_map[node] = Node(node.val)
    
    while queue:
        current = queue.popleft()
        for neighbor in current.neighbors:
            if neighbor not in clone_map:
                clone_map[neighbor] = Node(neighbor.val)
                queue.append(neighbor)
            clone_map[current].neighbors.append(clone_map[neighbor])
    
    return clone_map[node]


def course_schedule(num_courses, prerequisites):
    """
    PROBLEM: Can finish all courses (no cycles)
    PATTERN: Topological sort / DFS cycle detection
    TIME: O(V + E), SPACE: O(V + E)
    """
    from collections import defaultdict, deque
    
    # Build graph
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # BFS topological sort
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    count = 0
    
    while queue:
        node = queue.popleft()
        count += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return count == num_courses


# ============================================================================
# HEAPS - IMPORTANT
# ============================================================================

def top_k_frequent(nums, k):
    """
    PROBLEM: Top K frequent elements
    PATTERN: Hash map + heap
    TIME: O(n log k), SPACE: O(n)
    """
    from collections import Counter
    import heapq
    
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)


def find_kth_largest(nums, k):
    """
    PROBLEM: Kth largest element
    PATTERN: Min heap of size k
    TIME: O(n log k), SPACE: O(k)
    """
    import heapq
    
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]


def merge_k_sorted_lists(lists):
    """
    PROBLEM: Merge k sorted linked lists
    PATTERN: Min heap
    TIME: O(n log k), SPACE: O(k)
    """
    import heapq
    
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))
    
    return dummy.next


# ============================================================================
# BACKTRACKING - ESSENTIAL
# ============================================================================

def generate_parentheses(n):
    """
    PROBLEM: Generate all valid parentheses combinations
    PATTERN: Backtracking with constraints
    TIME: O(4^n / sqrt(n)), SPACE: O(n)
    """
    result = []
    
    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return
        
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)
    
    backtrack('', 0, 0)
    return result


def permute(nums):
    """
    PROBLEM: All permutations
    PATTERN: Backtracking
    TIME: O(n! * n), SPACE: O(n)
    """
    result = []
    
    def backtrack(current):
        if len(current) == len(nums):
            result.append(current[:])
            return
        
        for num in nums:
            if num not in current:
                current.append(num)
                backtrack(current)
                current.pop()
    
    backtrack([])
    return result


def subsets(nums):
    """
    PROBLEM: All subsets
    PATTERN: Backtracking
    TIME: O(2^n), SPACE: O(n)
    """
    result = []
    
    def backtrack(start, current):
        result.append(current[:])
        
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Critical DSA Problems...")
    
    # Arrays
    print("Two Sum:", two_sum([2, 7, 11, 15], 9))
    print("Longest Substring:", longest_substring_without_repeating("abcabcbb"))
    print("Max Subarray:", max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
    
    # DP
    print("Climb Stairs:", climb_stairs(5))
    print("Coin Change:", coin_change([1, 2, 5], 11))
    
    # Binary Search
    print("Binary Search:", binary_search([1, 2, 3, 4, 5], 3))
    
    # Stack
    print("Valid Parentheses:", valid_parentheses("()[]{}"))
    
    print("\n✅ All critical patterns covered!")
