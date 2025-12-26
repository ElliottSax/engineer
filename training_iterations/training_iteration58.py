def valid_parentheses(s):
    """Check if parentheses are valid."""
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    for c in s:
        if c in '([{':
            stack.append(c)
        elif c in ')]}':
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()

    return len(stack) == 0

def min_add_to_make_valid(s):
    """Minimum insertions to make valid."""
    open_count = close_count = 0

    for c in s:
        if c == '(':
            open_count += 1
        elif open_count > 0:
            open_count -= 1
        else:
            close_count += 1

    return open_count + close_count

def longest_valid_parentheses(s):
    """Length of longest valid parentheses substring."""
    stack = [-1]
    max_len = 0

    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])

    return max_len

def remove_duplicate_letters(s):
    """Remove duplicate letters to get smallest string."""
    last_occurrence = {c: i for i, c in enumerate(s)}
    stack = []
    seen = set()

    for i, c in enumerate(s):
        if c in seen:
            continue
        while stack and c < stack[-1] and last_occurrence[stack[-1]] > i:
            seen.remove(stack.pop())
        stack.append(c)
        seen.add(c)

    return ''.join(stack)

def decode_string(s):
    """Decode encoded string like 3[a2[c]]."""
    stack = []
    curr_num = 0
    curr_str = ""

    for c in s:
        if c.isdigit():
            curr_num = curr_num * 10 + int(c)
        elif c == '[':
            stack.append((curr_str, curr_num))
            curr_str = ""
            curr_num = 0
        elif c == ']':
            prev_str, num = stack.pop()
            curr_str = prev_str + curr_str * num
        else:
            curr_str += c

    return curr_str

def basic_calculator(s):
    """Basic calculator with +, -, parentheses."""
    stack = []
    result = 0
    num = 0
    sign = 1

    for c in s:
        if c.isdigit():
            num = num * 10 + int(c)
        elif c == '+':
            result += sign * num
            num = 0
            sign = 1
        elif c == '-':
            result += sign * num
            num = 0
            sign = -1
        elif c == '(':
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif c == ')':
            result += sign * num
            num = 0
            result *= stack.pop()  # sign
            result += stack.pop()  # previous result

    return result + sign * num

def evaluate_rpn(tokens):
    """Evaluate Reverse Polish Notation."""
    stack = []

    for token in tokens:
        if token in '+-*/':
            b, a = stack.pop(), stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:
                stack.append(int(a / b))  # truncate towards zero
        else:
            stack.append(int(token))

    return stack[0]

def daily_temperatures(temperatures):
    """Days until warmer temperature."""
    n = len(temperatures)
    result = [0] * n
    stack = []

    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)

    return result

def asteroid_collision(asteroids):
    """Simulate asteroid collision."""
    stack = []

    for a in asteroids:
        while stack and a < 0 < stack[-1]:
            if stack[-1] < -a:
                stack.pop()
                continue
            elif stack[-1] == -a:
                stack.pop()
            break
        else:
            stack.append(a)

    return stack

def simplify_path(path):
    """Simplify Unix file path."""
    stack = []

    for part in path.split('/'):
        if part == '..':
            if stack:
                stack.pop()
        elif part and part != '.':
            stack.append(part)

    return '/' + '/'.join(stack)

def exclusive_time_functions(n, logs):
    """Exclusive time of each function."""
    result = [0] * n
    stack = []
    prev_time = 0

    for log in logs:
        parts = log.split(':')
        func_id = int(parts[0])
        typ = parts[1]
        time = int(parts[2])

        if typ == 'start':
            if stack:
                result[stack[-1]] += time - prev_time
            stack.append(func_id)
            prev_time = time
        else:
            result[stack.pop()] += time - prev_time + 1
            prev_time = time + 1

    return result

def car_fleet(target, position, speed):
    """Count car fleets arriving at target."""
    cars = sorted(zip(position, speed), reverse=True)
    times = [(target - p) / s for p, s in cars]

    fleets = 0
    curr_time = 0

    for time in times:
        if time > curr_time:
            fleets += 1
            curr_time = time

    return fleets

def min_stack_operations():
    """Min stack with O(1) getMin."""
    stack = []
    min_stack = []

    def push(x):
        stack.append(x)
        if not min_stack or x <= min_stack[-1]:
            min_stack.append(x)

    def pop():
        if stack.pop() == min_stack[-1]:
            min_stack.pop()

    def get_min():
        return min_stack[-1] if min_stack else None

    def top():
        return stack[-1] if stack else None

    return push, pop, top, get_min

# Tests
tests = [
    ("valid", valid_parentheses("()[]{}"), True),
    ("valid_no", valid_parentheses("(]"), False),
    ("min_add", min_add_to_make_valid("())"), 1),
    ("min_add_2", min_add_to_make_valid("((("), 3),
    ("longest_valid", longest_valid_parentheses(")()())"), 4),
    ("remove_dup", remove_duplicate_letters("bcabc"), "abc"),
    ("remove_dup_2", remove_duplicate_letters("cbacdcbc"), "acdb"),
    ("decode", decode_string("3[a]2[bc]"), "aaabcbc"),
    ("decode_2", decode_string("3[a2[c]]"), "accaccacc"),
    ("calc", basic_calculator("(1+(4+5+2)-3)+(6+8)"), 23),
    ("rpn", evaluate_rpn(["2","1","+","3","*"]), 9),
    ("rpn_2", evaluate_rpn(["4","13","5","/","+"]), 6),
    ("daily_temp", daily_temperatures([73,74,75,71,69,72,76,73]), [1,1,4,2,1,1,0,0]),
    ("asteroid", asteroid_collision([5,10,-5]), [5,10]),
    ("asteroid_2", asteroid_collision([10,2,-5]), [10]),
    ("simplify", simplify_path("/a/./b/../../c/"), "/c"),
    ("exclusive", exclusive_time_functions(2, ["0:start:0","1:start:2","1:end:5","0:end:6"]), [3, 4]),
    ("car_fleet", car_fleet(12, [10,8,0,5,3], [2,4,1,1,3]), 3),
]

# Min stack test
push, pop, top, get_min = min_stack_operations()
push(-2)
push(0)
push(-3)
tests.append(("min_stack_1", get_min(), -3))
pop()
tests.append(("min_stack_2", top(), 0))
tests.append(("min_stack_3", get_min(), -2))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
