def max_profit_single(prices):
    """Best time to buy and sell stock - one transaction."""
    if not prices:
        return 0
    min_price = prices[0]
    max_profit = 0

    for price in prices[1:]:
        max_profit = max(max_profit, price - min_price)
        min_price = min(min_price, price)

    return max_profit

def max_profit_unlimited(prices):
    """Best time to buy and sell stock - unlimited transactions."""
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit

def max_profit_two_transactions(prices):
    """Best time to buy and sell stock - at most 2 transactions."""
    buy1 = buy2 = float('inf')
    profit1 = profit2 = 0

    for price in prices:
        buy1 = min(buy1, price)
        profit1 = max(profit1, price - buy1)
        buy2 = min(buy2, price - profit1)
        profit2 = max(profit2, price - buy2)

    return profit2

def max_profit_k_transactions(k, prices):
    """Best time to buy and sell stock - at most k transactions."""
    n = len(prices)
    if n == 0 or k == 0:
        return 0

    if k >= n // 2:
        return max_profit_unlimited(prices)

    dp = [[0] * n for _ in range(k + 1)]

    for t in range(1, k + 1):
        max_diff = -prices[0]
        for d in range(1, n):
            dp[t][d] = max(dp[t][d-1], prices[d] + max_diff)
            max_diff = max(max_diff, dp[t-1][d] - prices[d])

    return dp[k][n-1]

def max_profit_with_cooldown(prices):
    """Best time to buy and sell with cooldown."""
    if len(prices) < 2:
        return 0

    hold = -prices[0]  # holding stock
    sold = 0           # just sold
    rest = 0           # resting

    for price in prices[1:]:
        new_hold = max(hold, rest - price)
        new_sold = hold + price
        new_rest = max(rest, sold)
        hold, sold, rest = new_hold, new_sold, new_rest

    return max(sold, rest)

def max_profit_with_fee(prices, fee):
    """Best time to buy and sell with transaction fee."""
    hold = -prices[0]
    cash = 0

    for price in prices[1:]:
        hold = max(hold, cash - price)
        cash = max(cash, hold + price - fee)

    return cash

def house_robber(nums):
    """Maximum money without robbing adjacent houses."""
    if not nums:
        return 0
    if len(nums) <= 2:
        return max(nums)

    prev2, prev1 = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        prev2, prev1 = prev1, max(prev1, prev2 + nums[i])

    return prev1

def house_robber_circular(nums):
    """House robber with circular arrangement."""
    if len(nums) == 1:
        return nums[0]

    def rob_linear(arr):
        prev2, prev1 = 0, 0
        for num in arr:
            prev2, prev1 = prev1, max(prev1, prev2 + num)
        return prev1

    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))

def paint_house(costs):
    """Minimum cost to paint houses with 3 colors."""
    if not costs:
        return 0

    prev = costs[0][:]

    for i in range(1, len(costs)):
        curr = [
            costs[i][0] + min(prev[1], prev[2]),
            costs[i][1] + min(prev[0], prev[2]),
            costs[i][2] + min(prev[0], prev[1])
        ]
        prev = curr

    return min(prev)

def paint_fence(n, k):
    """Number of ways to paint fence with k colors."""
    if n == 0:
        return 0
    if n == 1:
        return k
    if n == 2:
        return k * k

    same = k
    diff = k * (k - 1)

    for _ in range(3, n + 1):
        same, diff = diff, (same + diff) * (k - 1)

    return same + diff

def decode_ways(s):
    """Number of ways to decode string to letters."""
    if not s or s[0] == '0':
        return 0

    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n + 1):
        if s[i-1] != '0':
            dp[i] = dp[i-1]
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]

    return dp[n]

def min_cost_climbing_stairs(cost):
    """Minimum cost to climb stairs."""
    n = len(cost)
    if n <= 1:
        return 0

    prev2, prev1 = cost[0], cost[1]
    for i in range(2, n):
        prev2, prev1 = prev1, cost[i] + min(prev1, prev2)

    return min(prev1, prev2)

def maximum_sum_circular_subarray(nums):
    """Maximum sum subarray in circular array."""
    def kadane(arr):
        max_sum = curr = arr[0]
        for num in arr[1:]:
            curr = max(num, curr + num)
            max_sum = max(max_sum, curr)
        return max_sum

    max_normal = kadane(nums)
    total = sum(nums)
    min_subarray = -kadane([-x for x in nums])

    if min_subarray == total:
        return max_normal

    return max(max_normal, total - min_subarray)

# Tests
tests = [
    ("stock1", max_profit_single([7,1,5,3,6,4]), 5),
    ("stock2", max_profit_unlimited([7,1,5,3,6,4]), 7),
    ("stock3", max_profit_two_transactions([3,3,5,0,0,3,1,4]), 6),
    ("stock_k", max_profit_k_transactions(2, [2,4,1]), 2),
    ("cooldown", max_profit_with_cooldown([1,2,3,0,2]), 3),
    ("fee", max_profit_with_fee([1,3,2,8,4,9], 2), 8),
    ("robber", house_robber([2,7,9,3,1]), 12),
    ("robber2", house_robber_circular([2,3,2]), 3),
    ("robber2_b", house_robber_circular([1,2,3,1]), 4),
    ("paint", paint_house([[17,2,17],[16,16,5],[14,3,19]]), 10),
    ("fence", paint_fence(3, 2), 6),
    ("decode", decode_ways("226"), 3),
    ("decode_2", decode_ways("12"), 2),
    ("stairs", min_cost_climbing_stairs([10,15,20]), 15),
    ("circular", maximum_sum_circular_subarray([5,-3,5]), 10),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
