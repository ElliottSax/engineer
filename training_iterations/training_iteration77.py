def reverse_string(s):
    """Reverse string in-place."""
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
    return s

def reverse_words(s):
    """Reverse words in string."""
    return ' '.join(s.split()[::-1])

def reverse_words_ii(s):
    """Reverse words in char array."""
    def reverse(arr, start, end):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1

    s.reverse()
    start = 0
    for end in range(len(s) + 1):
        if end == len(s) or s[end] == ' ':
            reverse(s, start, end - 1)
            start = end + 1
    return s

def string_compression(chars):
    """Compress string in-place."""
    write = read = 0

    while read < len(chars):
        char = chars[read]
        count = 0
        while read < len(chars) and chars[read] == char:
            read += 1
            count += 1
        chars[write] = char
        write += 1
        if count > 1:
            for c in str(count):
                chars[write] = c
                write += 1

    return write

def multiply_strings(num1, num2):
    """Multiply two strings."""
    if num1 == "0" or num2 == "0":
        return "0"

    m, n = len(num1), len(num2)
    result = [0] * (m + n)

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            mul = int(num1[i]) * int(num2[j])
            p1, p2 = i + j, i + j + 1
            total = mul + result[p2]
            result[p2] = total % 10
            result[p1] += total // 10

    result_str = ''.join(map(str, result))
    return result_str.lstrip('0') or '0'

def add_strings(num1, num2):
    """Add two strings."""
    result = []
    carry = 0
    i, j = len(num1) - 1, len(num2) - 1

    while i >= 0 or j >= 0 or carry:
        n1 = int(num1[i]) if i >= 0 else 0
        n2 = int(num2[j]) if j >= 0 else 0
        total = n1 + n2 + carry
        result.append(str(total % 10))
        carry = total // 10
        i -= 1
        j -= 1

    return ''.join(result[::-1])

def compare_version(version1, version2):
    """Compare two version numbers."""
    v1 = list(map(int, version1.split('.')))
    v2 = list(map(int, version2.split('.')))

    while len(v1) < len(v2):
        v1.append(0)
    while len(v2) < len(v1):
        v2.append(0)

    for a, b in zip(v1, v2):
        if a < b:
            return -1
        elif a > b:
            return 1
    return 0

def valid_number(s):
    """Check if string is valid number."""
    s = s.strip()
    if not s:
        return False

    seen_num = seen_dot = seen_e = False
    sign_ok = True

    for i, c in enumerate(s):
        if c.isdigit():
            seen_num = True
            sign_ok = False
        elif c in '+-':
            if not sign_ok:
                return False
            sign_ok = False
        elif c == '.':
            if seen_dot or seen_e:
                return False
            seen_dot = True
            sign_ok = False
        elif c in 'eE':
            if seen_e or not seen_num:
                return False
            seen_e = True
            seen_num = False
            sign_ok = True
        else:
            return False

    return seen_num

def roman_to_int(s):
    """Convert Roman numeral to integer."""
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    for i in range(len(s)):
        if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
            result -= values[s[i]]
        else:
            result += values[s[i]]
    return result

def int_to_roman(num):
    """Convert integer to Roman numeral."""
    values = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
              (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
              (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    result = []
    for val, symbol in values:
        while num >= val:
            result.append(symbol)
            num -= val
    return ''.join(result)

def atoi(s):
    """String to integer (atoi)."""
    s = s.strip()
    if not s:
        return 0

    sign = 1
    idx = 0
    if s[0] in '+-':
        sign = -1 if s[0] == '-' else 1
        idx = 1

    result = 0
    while idx < len(s) and s[idx].isdigit():
        result = result * 10 + int(s[idx])
        idx += 1

    result *= sign
    INT_MAX, INT_MIN = 2**31 - 1, -2**31
    return max(INT_MIN, min(INT_MAX, result))

def count_and_say(n):
    """Generate nth count-and-say sequence."""
    result = "1"
    for _ in range(n - 1):
        new_result = []
        count = 1
        for i in range(1, len(result)):
            if result[i] == result[i - 1]:
                count += 1
            else:
                new_result.extend([str(count), result[i - 1]])
                count = 1
        new_result.extend([str(count), result[-1]])
        result = ''.join(new_result)
    return result

# Tests
tests = [
    ("reverse", reverse_string(list("hello")), list("olleh")),
    ("rev_words", reverse_words("the sky is blue"), "blue is sky the"),
    ("rev_words_ii", reverse_words_ii(list("the sky")), list("sky the")),
    ("compress", string_compression(list("aabbccc")), 6),
    ("multiply", multiply_strings("123", "456"), "56088"),
    ("add", add_strings("123", "456"), "579"),
    ("compare", compare_version("1.01", "1.001"), 0),
    ("compare_2", compare_version("1.0", "1.0.0"), 0),
    ("compare_3", compare_version("0.1", "1.1"), -1),
    ("valid_num", valid_number("0"), True),
    ("valid_num_2", valid_number("e"), False),
    ("valid_num_3", valid_number("2e10"), True),
    ("roman_to", roman_to_int("MCMXCIV"), 1994),
    ("int_to_roman", int_to_roman(1994), "MCMXCIV"),
    ("atoi", atoi("   -42"), -42),
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
