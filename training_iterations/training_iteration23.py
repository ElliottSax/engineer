def atoi(s):
    """Converts string to integer (handles overflow, signs, whitespace)."""
    s = s.strip()
    if not s:
        return 0
    sign = 1
    i = 0
    if s[0] == '-':
        sign = -1
        i = 1
    elif s[0] == '+':
        i = 1
    result = 0
    INT_MAX, INT_MIN = 2**31 - 1, -2**31
    while i < len(s) and s[i].isdigit():
        result = result * 10 + int(s[i])
        i += 1
    result *= sign
    return max(INT_MIN, min(INT_MAX, result))

def multiply_strings(num1, num2):
    """Multiplies two numbers represented as strings."""
    if num1 == '0' or num2 == '0':
        return '0'
    m, n = len(num1), len(num2)
    result = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            mul = int(num1[i]) * int(num2[j])
            p1, p2 = i + j, i + j + 1
            total = mul + result[p2]
            result[p2] = total % 10
            result[p1] += total // 10
    result_str = ''.join(map(str, result)).lstrip('0')
    return result_str or '0'

def add_strings(num1, num2):
    """Adds two numbers represented as strings."""
    result = []
    carry = 0
    i, j = len(num1) - 1, len(num2) - 1
    while i >= 0 or j >= 0 or carry:
        x = int(num1[i]) if i >= 0 else 0
        y = int(num2[j]) if j >= 0 else 0
        total = x + y + carry
        result.append(str(total % 10))
        carry = total // 10
        i -= 1
        j -= 1
    return ''.join(reversed(result))

def compare_version(version1, version2):
    """Compares version strings."""
    v1 = list(map(int, version1.split('.')))
    v2 = list(map(int, version2.split('.')))
    n = max(len(v1), len(v2))
    for i in range(n):
        a = v1[i] if i < len(v1) else 0
        b = v2[i] if i < len(v2) else 0
        if a < b:
            return -1
        elif a > b:
            return 1
    return 0

def excel_column_number(column_title):
    """Converts Excel column title to number (A=1, B=2, ..., Z=26, AA=27)."""
    result = 0
    for char in column_title:
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result

def excel_column_title(n):
    """Converts number to Excel column title."""
    result = []
    while n > 0:
        n -= 1
        result.append(chr(n % 26 + ord('A')))
        n //= 26
    return ''.join(reversed(result))

def fraction_to_decimal(numerator, denominator):
    """Converts fraction to decimal string (handles repeating)."""
    if numerator == 0:
        return "0"
    result = []
    if (numerator < 0) != (denominator < 0):
        result.append('-')
    numerator, denominator = abs(numerator), abs(denominator)
    result.append(str(numerator // denominator))
    remainder = numerator % denominator
    if remainder == 0:
        return ''.join(result)
    result.append('.')
    seen = {}
    while remainder != 0:
        if remainder in seen:
            result.insert(seen[remainder], '(')
            result.append(')')
            break
        seen[remainder] = len(result)
        remainder *= 10
        result.append(str(remainder // denominator))
        remainder %= denominator
    return ''.join(result)

def integer_to_roman(num):
    """Converts integer to Roman numeral."""
    values = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
              (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
              (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    result = []
    for value, numeral in values:
        while num >= value:
            result.append(numeral)
            num -= value
    return ''.join(result)

def roman_to_integer(s):
    """Converts Roman numeral to integer."""
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    for i in range(len(s)):
        if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
            result -= values[s[i]]
        else:
            result += values[s[i]]
    return result

def sqrt_integer(x):
    """Integer square root using binary search."""
    if x < 2:
        return x
    left, right = 1, x // 2
    while left <= right:
        mid = (left + right) // 2
        if mid * mid == x:
            return mid
        elif mid * mid < x:
            left = mid + 1
        else:
            right = mid - 1
    return right

def pow_x_n(x, n):
    """Calculates x^n using fast exponentiation."""
    if n < 0:
        x = 1 / x
        n = -n
    result = 1
    while n:
        if n & 1:
            result *= x
        x *= x
        n >>= 1
    return result

def reverse_integer(x):
    """Reverses integer digits (handles overflow)."""
    INT_MAX, INT_MIN = 2**31 - 1, -2**31
    sign = 1 if x >= 0 else -1
    x = abs(x)
    result = 0
    while x:
        result = result * 10 + x % 10
        x //= 10
    result *= sign
    return result if INT_MIN <= result <= INT_MAX else 0

def count_and_say(n):
    """Generates nth term of count-and-say sequence."""
    result = "1"
    for _ in range(n - 1):
        next_result = []
        i = 0
        while i < len(result):
            char = result[i]
            count = 1
            while i + count < len(result) and result[i + count] == char:
                count += 1
            next_result.append(str(count) + char)
            i += count
        result = ''.join(next_result)
    return result

# Tests
tests = [
    ("atoi_basic", atoi("42"), 42),
    ("atoi_negative", atoi("   -42"), -42),
    ("atoi_overflow", atoi("2147483648"), 2147483647),
    ("atoi_words", atoi("4193 with words"), 4193),
    ("multiply", multiply_strings("123", "456"), "56088"),
    ("multiply_zero", multiply_strings("0", "456"), "0"),
    ("add_strings", add_strings("11", "123"), "134"),
    ("compare_less", compare_version("1.01", "1.001"), 0),
    ("compare_greater", compare_version("1.0.1", "1"), 1),
    ("excel_num", excel_column_number("AB"), 28),
    ("excel_num_2", excel_column_number("ZY"), 701),
    ("excel_title", excel_column_title(28), "AB"),
    ("excel_title_2", excel_column_title(701), "ZY"),
    ("fraction", fraction_to_decimal(1, 2), "0.5"),
    ("fraction_repeat", fraction_to_decimal(4, 333), "0.(012)"),
    ("int_to_roman", integer_to_roman(1994), "MCMXCIV"),
    ("roman_to_int", roman_to_integer("MCMXCIV"), 1994),
    ("sqrt", sqrt_integer(8), 2),
    ("sqrt_perfect", sqrt_integer(16), 4),
    ("pow", pow_x_n(2.0, 10), 1024.0),
    ("pow_neg", pow_x_n(2.0, -2), 0.25),
    ("reverse", reverse_integer(123), 321),
    ("reverse_neg", reverse_integer(-123), -321),
    ("count_say", count_and_say(4), "1211"),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
