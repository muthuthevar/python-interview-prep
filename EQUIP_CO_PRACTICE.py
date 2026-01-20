"""
EQUIP.CO ASSESSMENT - Most Likely Coding Problems
Practice these problems - they're commonly asked in Equip.co Python assessments
"""


# ============================================================================
# PROBLEM 1: Two Sum (VERY COMMON)
# ============================================================================
def two_sum(nums, target):
    """
    Given an array of integers nums and an integer target,
    return indices of the two numbers such that they add up to target.

    Time: O(n), Space: O(n)
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


# ============================================================================
# PROBLEM 2: Longest Substring Without Repeating Characters
# ============================================================================
def length_of_longest_substring(s):
    """
    Find the length of the longest substring without repeating characters.

    Time: O(n), Space: O(min(n, m)) where m is charset size
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


# ============================================================================
# PROBLEM 3: Valid Parentheses
# ============================================================================
def is_valid(s):
    """
    Determine if the input string is valid (matching brackets).

    Time: O(n), Space: O(n)
    """
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}

    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)

    return not stack


# ============================================================================
# PROBLEM 4: Merge Two Sorted Lists
# ============================================================================
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_two_lists(l1, l2):
    """
    Merge two sorted linked lists and return the merged list.

    Time: O(n + m), Space: O(1)
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


# ============================================================================
# PROBLEM 5: Maximum Subarray (Kadane's Algorithm)
# ============================================================================
def max_subarray(nums):
    """
    Find the contiguous subarray with the largest sum.

    Time: O(n), Space: O(1)
    """
    max_sum = current_sum = nums[0]

    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum


# ============================================================================
# PROBLEM 6: Binary Search
# ============================================================================
def search(nums, target):
    """
    Search target in sorted array. Return index or -1.

    Time: O(log n), Space: O(1)
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


# ============================================================================
# PROBLEM 7: Group Anagrams
# ============================================================================
def group_anagrams(strs):
    """
    Group strings that are anagrams of each other.

    Time: O(n * k log k) where k is max string length, Space: O(n * k)
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for s in strs:
        key = "".join(sorted(s))
        groups[key].append(s)

    return list(groups.values())


# ============================================================================
# PROBLEM 8: Climbing Stairs (Fibonacci Pattern)
# ============================================================================
def climb_stairs(n):
    """
    You are climbing a staircase. It takes n steps to reach the top.
    Each time you can either climb 1 or 2 steps. In how many ways can you climb?

    Time: O(n), Space: O(1)
    """
    if n <= 2:
        return n

    first, second = 1, 2
    for i in range(3, n + 1):
        first, second = second, first + second

    return second


# ============================================================================
# PROBLEM 9: Product of Array Except Self
# ============================================================================
def product_except_self(nums):
    """
    Return an array such that answer[i] is product of all elements except nums[i].
    Cannot use division.

    Time: O(n), Space: O(1) excluding output array
    """
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


# ============================================================================
# PROBLEM 10: Container With Most Water
# ============================================================================
def max_area(height):
    """
    Find two lines that together with x-axis forms a container with most water.

    Time: O(n), Space: O(1)
    """
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
# PROBLEM 11: Reverse Linked List
# ============================================================================
def reverse_list(head):
    """
    Reverse a singly linked list.

    Time: O(n), Space: O(1)
    """
    prev = None
    current = head

    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp

    return prev


# ============================================================================
# PROBLEM 12: Contains Duplicate
# ============================================================================
def contains_duplicate(nums):
    """
    Return true if any value appears at least twice in the array.

    Time: O(n), Space: O(n)
    """
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False


# ============================================================================
# PROBLEM 13: Best Time to Buy and Sell Stock
# ============================================================================
def max_profit(prices):
    """
    Find the maximum profit from buying and selling stock once.

    Time: O(n), Space: O(1)
    """
    if not prices:
        return 0

    min_price = prices[0]
    max_profit = 0

    for price in prices[1:]:
        max_profit = max(max_profit, price - min_price)
        min_price = min(min_price, price)

    return max_profit


# ============================================================================
# PROBLEM 14: Valid Anagram
# ============================================================================
def is_anagram(s, t):
    """
    Determine if t is an anagram of s.

    Time: O(n), Space: O(1) - fixed size alphabet
    """
    if len(s) != len(t):
        return False

    from collections import Counter

    return Counter(s) == Counter(t)


# ============================================================================
# PROBLEM 15: Maximum Depth of Binary Tree
# ============================================================================
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root):
    """
    Find the maximum depth of a binary tree.

    Time: O(n), Space: O(h) where h is height
    """
    if not root:
        return 0

    return 1 + max(max_depth(root.left), max_depth(root.right))


# ============================================================================
# PROBLEM 16: Palindrome Number
# ============================================================================
def is_palindrome(x):
    """
    Determine if integer is a palindrome without converting to string.

    Time: O(log n), Space: O(1)
    """
    if x < 0 or (x % 10 == 0 and x != 0):
        return False

    reversed_num = 0
    original = x

    while x > reversed_num:
        reversed_num = reversed_num * 10 + x % 10
        x //= 10

    return x == reversed_num or x == reversed_num // 10


# ============================================================================
# PROBLEM 17: Rotate Array
# ============================================================================
def rotate(nums, k):
    """
    Rotate array to the right by k steps (in-place).

    Time: O(n), Space: O(1)
    """
    n = len(nums)
    k = k % n

    # Reverse entire array
    nums.reverse()
    # Reverse first k elements
    nums[:k] = reversed(nums[:k])
    # Reverse remaining elements
    nums[k:] = reversed(nums[k:])


# ============================================================================
# PROBLEM 18: Top K Frequent Elements
# ============================================================================
def top_k_frequent(nums, k):
    """
    Return the k most frequent elements.

    Time: O(n log k), Space: O(n)
    """
    from collections import Counter
    import heapq

    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)


# ============================================================================
# PROBLEM 19: Missing Number
# ============================================================================
def missing_number(nums):
    """
    Find the missing number in array containing n distinct numbers in range [0, n].

    Time: O(n), Space: O(1)
    """
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum


# ============================================================================
# PROBLEM 20: Single Number
# ============================================================================
def single_number(nums):
    """
    Find the single number that appears once (others appear twice).
    Use XOR property: a ^ a = 0, a ^ 0 = a

    Time: O(n), Space: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    # Test cases
    print("Testing Two Sum:", two_sum([2, 7, 11, 15], 9))  # [0, 1]
    print("Testing Longest Substring:", length_of_longest_substring("abcabcbb"))  # 3
    print("Testing Valid Parentheses:", is_valid("()[]{}"))  # True
    print("Testing Max Subarray:", max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  # 6
    print("Testing Binary Search:", search([1, 2, 3, 4, 5], 3))  # 2
    print("Testing Climb Stairs:", climb_stairs(5))  # 8
    print("Testing Contains Duplicate:", contains_duplicate([1, 2, 3, 1]))  # True
    print("Testing Max Profit:", max_profit([7, 1, 5, 3, 6, 4]))  # 5
    print("Testing Is Anagram:", is_anagram("anagram", "nagaram"))  # True
    print("Testing Missing Number:", missing_number([3, 0, 1]))  # 2
    print("Testing Single Number:", single_number([4, 1, 2, 1, 2]))  # 4
