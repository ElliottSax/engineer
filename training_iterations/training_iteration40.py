def basic_calculator_ii(s):
    """Evaluates expression with +, -, *, /."""
    stack = []
    num = 0
    sign = '+'
    s = s.replace(' ', '') + '+'
    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        else:
            if sign == '+':
                stack.append(num)
            elif sign == '-':
                stack.append(-num)
            elif sign == '*':
                stack.append(stack.pop() * num)
            elif sign == '/':
                stack.append(int(stack.pop() / num))
            num = 0
            sign = char
    return sum(stack)

def expression_add_operators(num, target):
    """All expressions with +, -, * that evaluate to target."""
    result = []

    def backtrack(idx, path, value, prev):
        if idx == len(num):
            if value == target:
                result.append(path)
            return
        for i in range(idx, len(num)):
            if i > idx and num[idx] == '0':
                break
            curr_str = num[idx:i+1]
            curr_num = int(curr_str)
            if idx == 0:
                backtrack(i + 1, curr_str, curr_num, curr_num)
            else:
                backtrack(i + 1, path + '+' + curr_str, value + curr_num, curr_num)
                backtrack(i + 1, path + '-' + curr_str, value - curr_num, -curr_num)
                backtrack(i + 1, path + '*' + curr_str, value - prev + prev * curr_num, prev * curr_num)

    backtrack(0, '', 0, 0)
    return result

def different_ways_to_add_parentheses(expression):
    """All possible results from adding parentheses."""
    if expression.isdigit():
        return [int(expression)]
    results = []
    for i, char in enumerate(expression):
        if char in '+-*':
            left = different_ways_to_add_parentheses(expression[:i])
            right = different_ways_to_add_parentheses(expression[i+1:])
            for l in left:
                for r in right:
                    if char == '+':
                        results.append(l + r)
                    elif char == '-':
                        results.append(l - r)
                    else:
                        results.append(l * r)
    return results

def calculate_with_parentheses(s):
    """Full calculator with +, -, *, /, ()."""
    def helper(tokens, idx):
        stack = []
        num = 0
        sign = '+'
        while idx < len(tokens):
            token = tokens[idx]
            if token.isdigit():
                num = int(token)
            elif token == '(':
                num, idx = helper(tokens, idx + 1)
            elif token == ')':
                break
            if token in '+-*/' or idx == len(tokens) - 1 or token == ')':
                if sign == '+':
                    stack.append(num)
                elif sign == '-':
                    stack.append(-num)
                elif sign == '*':
                    stack.append(stack.pop() * num)
                elif sign == '/':
                    stack.append(int(stack.pop() / num))
                sign = token
                num = 0
            idx += 1
        return sum(stack), idx

    # Tokenize
    tokens = []
    i = 0
    while i < len(s):
        if s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(s[i:j])
            i = j
        elif s[i] != ' ':
            tokens.append(s[i])
            i += 1
        else:
            i += 1
    return helper(tokens, 0)[0]

def evaluate_ternary(expression):
    """Evaluates ternary expression."""
    stack = []
    for i in range(len(expression) - 1, -1, -1):
        char = expression[i]
        if stack and stack[-1] == '?':
            stack.pop()  # ?
            first = stack.pop()
            stack.pop()  # :
            second = stack.pop()
            stack.append(first if char == 'T' else second)
        else:
            stack.append(char)
    return stack[0]

def parse_lisp_expression(expression):
    """Evaluates Lisp-like expression."""
    def parse(tokens, idx):
        token = tokens[idx]
        if token != '(':
            # Variable or number
            try:
                return int(token), idx + 1
            except ValueError:
                return token, idx + 1
        # Parse list
        idx += 1  # Skip (
        op = tokens[idx]
        idx += 1
        if op == 'let':
            bindings = []
            while tokens[idx] != ')':
                if tokens[idx + 1] == ')' or tokens[idx + 1] == '(':
                    # Last expression
                    break
                var = tokens[idx]
                idx += 1
                val, idx = parse(tokens, idx)
                bindings.append((var, val))
            # Evaluate body
            result, idx = parse(tokens, idx)
            idx += 1  # Skip )
            return result, idx
        elif op in ('add', 'mult'):
            left, idx = parse(tokens, idx)
            right, idx = parse(tokens, idx)
            idx += 1  # Skip )
            if op == 'add':
                return left + right, idx
            else:
                return left * right, idx
        return None, idx

    # Tokenize
    tokens = []
    i = 0
    while i < len(expression):
        if expression[i] in '()':
            tokens.append(expression[i])
            i += 1
        elif expression[i] == ' ':
            i += 1
        else:
            j = i
            while j < len(expression) and expression[j] not in '() ':
                j += 1
            tokens.append(expression[i:j])
            i = j

    result, _ = parse(tokens, 0)
    return result

def valid_number(s):
    """Checks if string is valid number."""
    s = s.strip()
    if not s:
        return False
    seen_digit = seen_dot = seen_e = False
    for i, c in enumerate(s):
        if c.isdigit():
            seen_digit = True
        elif c in '+-':
            if i != 0 and s[i-1].lower() != 'e':
                return False
        elif c == '.':
            if seen_dot or seen_e:
                return False
            seen_dot = True
        elif c.lower() == 'e':
            if seen_e or not seen_digit:
                return False
            seen_e = True
            seen_digit = False
        else:
            return False
    return seen_digit

# Tests
tests = [
    ("calc_ii", basic_calculator_ii("3+2*2"), 7),
    ("calc_ii_2", basic_calculator_ii(" 3/2 "), 1),
    ("calc_ii_3", basic_calculator_ii(" 3+5 / 2 "), 5),
    ("expr_add_ops", sorted(expression_add_operators("123", 6)), sorted(["1+2+3", "1*2*3"])),
    ("expr_add_ops_2", len(expression_add_operators("105", 5)), 2),
    ("diff_ways", sorted(different_ways_to_add_parentheses("2-1-1")), [-2, 0, 2]),
    ("ternary", evaluate_ternary("T?2:3"), "2"),
    ("ternary_nested", evaluate_ternary("T?T?F:5:3"), "F"),
    ("valid_num", valid_number("0"), True),
    ("valid_num_2", valid_number(" 0.1"), True),
    ("valid_num_3", valid_number("abc"), False),
    ("valid_num_4", valid_number("2e10"), True),
    ("valid_num_5", valid_number("-90e3"), True),
    ("valid_num_6", valid_number("1e"), False),
]

# Lisp expression test
tests.append(("lisp_add", parse_lisp_expression("(add 1 2)"), 3))
tests.append(("lisp_mult", parse_lisp_expression("(mult 3 (add 2 3))"), 15))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
