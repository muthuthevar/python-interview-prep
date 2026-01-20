"""
Practice Coding Problems for Python Interview
Solutions to common interview problems
"""

# ============================================================================
# ARRAYS & STRINGS
# ============================================================================


def two_sum(nums, target):
    """Find two numbers that add up to target. Return indices."""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


def three_sum(nums):
    """Find all unique triplets that sum to zero."""
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    return result


def longest_substring_without_repeating(s):
    """Find length of longest substring without repeating characters."""
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
    """Group strings that are anagrams of each other."""
    from collections import defaultdict

    groups = defaultdict(list)
    for s in strs:
        key = "".join(sorted(s))
        groups[key].append(s)
    return list(groups.values())


def rotate_array(nums, k):
    """Rotate array to the right by k steps (in-place)."""
    n = len(nums)
    k = k % n
    nums[:] = nums[n - k :] + nums[: n - k]


# ============================================================================
# LINKED LISTS
# ============================================================================


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_linked_list(head):
    """Reverse a linked list iteratively."""
    prev = None
    current = head
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    return prev


def merge_two_sorted_lists(l1, l2):
    """Merge two sorted linked lists."""
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
    """Detect if linked list has a cycle (Floyd's algorithm)."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


def remove_nth_from_end(head, n):
    """Remove nth node from end of list."""
    dummy = ListNode(0)
    dummy.next = head
    first = second = dummy

    for _ in range(n + 1):
        first = first.next

    while first:
        first = first.next
        second = second.next

    second.next = second.next.next
    return dummy.next


# ============================================================================
# TREES
# ============================================================================


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_depth_binary_tree(root):
    """Find maximum depth of binary tree."""
    if not root:
        return 0
    return 1 + max(max_depth_binary_tree(root.left), max_depth_binary_tree(root.right))


def same_tree(p, q):
    """Check if two binary trees are the same."""
    if not p and not q:
        return True
    if not p or not q:
        return False
    return p.val == q.val and same_tree(p.left, q.left) and same_tree(p.right, q.right)


def invert_binary_tree(root):
    """Invert a binary tree."""
    if not root:
        return None
    root.left, root.right = root.right, root.left
    invert_binary_tree(root.left)
    invert_binary_tree(root.right)
    return root


def level_order_traversal(root):
    """Binary tree level order traversal (BFS)."""
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


def validate_bst(root, min_val=float("-inf"), max_val=float("inf")):
    """Validate if binary tree is a valid BST."""
    if not root:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return validate_bst(root.left, min_val, root.val) and validate_bst(
        root.right, root.val, max_val
    )


# ============================================================================
# STACKS & QUEUES
# ============================================================================


def valid_parentheses(s):
    """Check if parentheses are valid."""
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    return not stack


def daily_temperatures(temperatures):
    """Find number of days until warmer temperature (monotonic stack)."""
    stack = []
    result = [0] * len(temperatures)
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        stack.append(i)
    return result


# ============================================================================
# DYNAMIC PROGRAMMING
# ============================================================================


def climb_stairs(n):
    """Number of ways to climb n stairs (1 or 2 steps at a time)."""
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


def house_robber(nums):
    """Maximum money that can be robbed (no adjacent houses)."""
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

    return dp[-1]


def coin_change(coins, amount):
    """Minimum coins needed to make amount."""
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float("inf") else -1


def longest_common_subsequence(text1, text2):
    """Length of longest common subsequence."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


# ============================================================================
# GREEDY ALGORITHMS
# ============================================================================


def max_subarray(nums):
    """Maximum sum of contiguous subarray (Kadane's algorithm)."""
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum


def jump_game(nums):
    """Can you reach the last index?"""
    max_reach = 0
    for i, num in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + num)
        if max_reach >= len(nums) - 1:
            return True
    return True


# ============================================================================
# BINARY SEARCH
# ============================================================================


def binary_search(arr, target):
    """Binary search in sorted array."""
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
    """Search in rotated sorted array."""
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


# ============================================================================
# GRAPHS
# ============================================================================


def num_islands(grid):
    """Number of islands (DFS)."""
    if not grid:
        return 0

    def dfs(i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != "1":
            return
        grid[i][j] = "0"
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "1":
                dfs(i, j)
                count += 1
    return count


def clone_graph(node):
    """Clone an undirected graph."""
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


# ============================================================================
# HEAPS
# ============================================================================


def top_k_frequent(nums, k):
    """Top K frequent elements."""
    from collections import Counter
    import heapq

    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)


def find_kth_largest(nums, k):
    """Find Kth largest element."""
    import heapq

    return heapq.nlargest(k, nums)[-1]


# ============================================================================
# STRING MANIPULATION
# ============================================================================


def longest_palindromic_substring(s):
    """Find longest palindromic substring."""

    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1 : right]

    longest = ""
    for i in range(len(s)):
        palindrome1 = expand_around_center(i, i)
        palindrome2 = expand_around_center(i, i + 1)
        longest = max(longest, palindrome1, palindrome2, key=len)
    return longest


def valid_palindrome(s):
    """Check if string is palindrome (alphanumeric only)."""
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def product_except_self(nums):
    """Product of array except self (no division)."""
    n = len(nums)
    result = [1] * n

    # Left pass
    for i in range(1, n):
        result[i] = result[i - 1] * nums[i - 1]

    # Right pass
    right = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right
        right *= nums[i]

    return result


def container_with_most_water(height):
    """Container with most water (two pointers)."""
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


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test examples
    print("Two Sum:", two_sum([2, 7, 11, 15], 9))
    print("Longest Substring:", longest_substring_without_repeating("abcabcbb"))
    print("Valid Parentheses:", valid_parentheses("()[]{}"))
    print("Climb Stairs:", climb_stairs(5))
    print("Max Subarray:", max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
    print("Binary Search:", binary_search([1, 2, 3, 4, 5], 3))
