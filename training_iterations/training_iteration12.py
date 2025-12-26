def longest_increasing_subsequence(nums):
    """Length of longest increasing subsequence using binary search."""
    from bisect import bisect_left
    if not nums:
        return 0
    tails = []
    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)

def find_kth_largest(nums, k):
    """Finds kth largest element using quickselect."""
    import random
    def quickselect(left, right, k_smallest):
        if left == right:
            return nums[left]
        pivot_idx = random.randint(left, right)
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        pivot = nums[right]
        store_idx = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[store_idx], nums[i] = nums[i], nums[store_idx]
                store_idx += 1
        nums[store_idx], nums[right] = nums[right], nums[store_idx]
        if store_idx == k_smallest:
            return nums[store_idx]
        elif store_idx < k_smallest:
            return quickselect(store_idx + 1, right, k_smallest)
        else:
            return quickselect(left, store_idx - 1, k_smallest)
    return quickselect(0, len(nums) - 1, len(nums) - k)

def valid_palindrome_ii(s):
    """Can string become palindrome by removing at most one character?"""
    def is_palindrome(left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
        left += 1
        right -= 1
    return True

def product_except_self(nums):
    """Product of array except self without division."""
    n = len(nums)
    result = [1] * n
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    return result

def rotate_image(matrix):
    """Rotates NxN matrix 90 degrees clockwise in-place."""
    n = len(matrix)
    # Transpose
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Reverse rows
    for i in range(n):
        matrix[i].reverse()
    return matrix

def subsets(nums):
    """Generates all subsets of array."""
    result = [[]]
    for num in nums:
        result += [subset + [num] for subset in result]
    return result

def permutations(nums):
    """Generates all permutations."""
    if len(nums) <= 1:
        return [nums[:]]
    result = []
    for i, num in enumerate(nums):
        rest = nums[:i] + nums[i+1:]
        for perm in permutations(rest):
            result.append([num] + perm)
    return result

def combination_sum(candidates, target):
    """Finds combinations that sum to target (can reuse elements)."""
    result = []
    def backtrack(start, target, path):
        if target == 0:
            result.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] <= target:
                path.append(candidates[i])
                backtrack(i, target - candidates[i], path)
                path.pop()
    backtrack(0, target, [])
    return result

def letter_combinations(digits):
    """Phone keypad letter combinations."""
    if not digits:
        return []
    mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
               '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
    result = ['']
    for digit in digits:
        result = [prefix + char for prefix in result for char in mapping[digit]]
    return result

def generate_parentheses(n):
    """Generates all valid n pairs of parentheses."""
    result = []
    def backtrack(s, open_count, close_count):
        if len(s) == 2 * n:
            result.append(s)
            return
        if open_count < n:
            backtrack(s + '(', open_count + 1, close_count)
        if close_count < open_count:
            backtrack(s + ')', open_count, close_count + 1)
    backtrack('', 0, 0)
    return result

def next_permutation(nums):
    """Rearranges to next lexicographically greater permutation."""
    n = len(nums)
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i >= 0:
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[i+1:] = reversed(nums[i+1:])
    return nums

# Tests
tests = [
    ("lis", longest_increasing_subsequence([10,9,2,5,3,7,101,18]), 4),
    ("kth_largest", find_kth_largest([3,2,1,5,6,4], 2), 5),
    ("valid_palindrome_ii", valid_palindrome_ii("abca"), True),
    ("valid_palindrome_ii_no", valid_palindrome_ii("abc"), False),
    ("product_except", product_except_self([1,2,3,4]), [24,12,8,6]),
    ("rotate", rotate_image([[1,2],[3,4]]), [[3,1],[4,2]]),
    ("subsets", len(subsets([1,2,3])), 8),
    ("permutations", len(permutations([1,2,3])), 6),
    ("comb_sum", sorted([sorted(x) for x in combination_sum([2,3,6,7], 7)]), [[2,2,3],[7]]),
    ("letter_comb", sorted(letter_combinations("23")), sorted(["ad","ae","af","bd","be","bf","cd","ce","cf"])),
    ("gen_parens", len(generate_parentheses(3)), 5),
    ("next_perm", next_permutation([1,2,3]), [1,3,2]),
    ("next_perm_desc", next_permutation([3,2,1]), [1,2,3]),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
