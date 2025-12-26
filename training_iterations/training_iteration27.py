def design_twitter():
    """Design Twitter with follow, tweet, news feed."""
    from collections import defaultdict
    import heapq

    tweets = defaultdict(list)  # userId -> [(time, tweetId), ...]
    following = defaultdict(set)  # userId -> set of followeeIds
    time = [0]

    def post_tweet(user_id, tweet_id):
        tweets[user_id].append((time[0], tweet_id))
        time[0] += 1

    def get_news_feed(user_id):
        heap = []
        users = following[user_id] | {user_id}
        for uid in users:
            if tweets[uid]:
                idx = len(tweets[uid]) - 1
                t, tid = tweets[uid][idx]
                heapq.heappush(heap, (-t, tid, uid, idx))
        result = []
        while heap and len(result) < 10:
            _, tid, uid, idx = heapq.heappop(heap)
            result.append(tid)
            if idx > 0:
                t, next_tid = tweets[uid][idx - 1]
                heapq.heappush(heap, (-t, next_tid, uid, idx - 1))
        return result

    def follow(follower_id, followee_id):
        if follower_id != followee_id:
            following[follower_id].add(followee_id)

    def unfollow(follower_id, followee_id):
        following[follower_id].discard(followee_id)

    return post_tweet, get_news_feed, follow, unfollow

def design_file_system():
    """Design file system with create path and get value."""
    paths = {'/': -1}

    def create_path(path, value):
        if path in paths or path == '/':
            return False
        parent = path.rsplit('/', 1)[0]
        if parent == '':
            parent = '/'
        if parent not in paths:
            return False
        paths[path] = value
        return True

    def get(path):
        return paths.get(path, -1)

    return create_path, get

def design_browser_history():
    """Browser history with visit, back, forward."""
    history = []
    current = [-1]

    def visit(url):
        current[0] += 1
        history[current[0]:] = [url]

    def back(steps):
        current[0] = max(0, current[0] - steps)
        return history[current[0]]

    def forward(steps):
        current[0] = min(len(history) - 1, current[0] + steps)
        return history[current[0]]

    def init(homepage):
        history.append(homepage)
        current[0] = 0

    return init, visit, back, forward

def design_hit_counter():
    """Hit counter with hits in last 5 minutes."""
    from collections import deque
    hits = deque()

    def hit(timestamp):
        hits.append(timestamp)

    def get_hits(timestamp):
        while hits and hits[0] <= timestamp - 300:
            hits.popleft()
        return len(hits)

    return hit, get_hits

def design_underground():
    """Underground system tracking travel times."""
    from collections import defaultdict
    check_ins = {}  # id -> (station, time)
    travel_times = defaultdict(lambda: [0, 0])  # (start, end) -> [total_time, count]

    def check_in(cid, station, time):
        check_ins[cid] = (station, time)

    def check_out(cid, station, time):
        start_station, start_time = check_ins[cid]
        key = (start_station, station)
        travel_times[key][0] += time - start_time
        travel_times[key][1] += 1
        del check_ins[cid]

    def get_average(start, end):
        total, count = travel_times[(start, end)]
        return total / count

    return check_in, check_out, get_average

def design_logger():
    """Rate limiter logger."""
    messages = {}

    def should_print(timestamp, message):
        if message not in messages or timestamp - messages[message] >= 10:
            messages[message] = timestamp
            return True
        return False

    return should_print

def design_parking():
    """Parking system with different sizes."""
    def init(big, medium, small):
        slots = [0, big, medium, small]

        def add_car(car_type):
            if slots[car_type] > 0:
                slots[car_type] -= 1
                return True
            return False

        return add_car

    return init

def design_leaderboard():
    """Leaderboard with add, top, reset."""
    from collections import defaultdict
    import heapq
    scores = defaultdict(int)

    def add_score(player_id, score):
        scores[player_id] += score

    def top(k):
        return sum(heapq.nlargest(k, scores.values()))

    def reset(player_id):
        scores[player_id] = 0

    return add_score, top, reset

def design_atm():
    """ATM machine."""
    denominations = [20, 50, 100, 200, 500]
    bank = [0] * 5

    def deposit(counts):
        for i in range(5):
            bank[i] += counts[i]

    def withdraw(amount):
        result = [0] * 5
        for i in range(4, -1, -1):
            need = min(bank[i], amount // denominations[i])
            result[i] = need
            amount -= need * denominations[i]
        if amount > 0:
            return [-1]
        for i in range(5):
            bank[i] -= result[i]
        return result

    return deposit, withdraw

# Tests
tests = []

# Twitter tests
post, feed, follow_user, unfollow = design_twitter()
post(1, 5)
tests.append(("twitter_feed", feed(1), [5]))
follow_user(1, 2)
post(2, 6)
tests.append(("twitter_follow", feed(1), [6, 5]))
unfollow(1, 2)
tests.append(("twitter_unfollow", feed(1), [5]))

# File system tests
create, get_path = design_file_system()
tests.append(("fs_create", create("/a", 1), True))
tests.append(("fs_get", get_path("/a"), 1))
tests.append(("fs_invalid", create("/a/b/c", 2), False))
tests.append(("fs_valid", create("/a/b", 2), True))

# Browser tests
init_browser, visit, back, forward = design_browser_history()
init_browser("google.com")
visit("facebook.com")
visit("youtube.com")
tests.append(("browser_back", back(1), "facebook.com"))
tests.append(("browser_forward", forward(1), "youtube.com"))
visit("linkedin.com")
tests.append(("browser_forward_blocked", forward(2), "linkedin.com"))

# Hit counter tests
hit, get_hits = design_hit_counter()
hit(1); hit(2); hit(3)
tests.append(("hit_count", get_hits(4), 3))
hit(300)
tests.append(("hit_expire", get_hits(300), 4))
tests.append(("hit_expire_2", get_hits(301), 3))

# Underground tests
check_in, check_out, get_avg = design_underground()
check_in(1, "A", 3)
check_out(1, "B", 8)
tests.append(("underground_avg", get_avg("A", "B"), 5.0))
check_in(2, "A", 10)
check_out(2, "B", 16)
tests.append(("underground_avg_2", get_avg("A", "B"), 5.5))

# Logger tests
should_print = design_logger()
tests.append(("logger_first", should_print(1, "foo"), True))
tests.append(("logger_dup", should_print(2, "foo"), False))
tests.append(("logger_ok", should_print(11, "foo"), True))

# Parking tests
init_parking = design_parking()
add_car = init_parking(1, 1, 0)
tests.append(("parking_big", add_car(1), True))
tests.append(("parking_medium", add_car(2), True))
tests.append(("parking_full", add_car(1), False))

# Leaderboard tests
add_score, top_k, reset_score = design_leaderboard()
add_score(1, 73)
add_score(2, 56)
add_score(3, 39)
add_score(4, 51)
tests.append(("leader_top3", top_k(3), 180))
reset_score(1)
tests.append(("leader_reset", top_k(3), 146))

# ATM tests
deposit, withdraw = design_atm()
deposit([0, 0, 1, 2, 1])  # 100 + 400 + 500 = 1000
tests.append(("atm_withdraw", withdraw(600), [0, 0, 1, 0, 1]))
tests.append(("atm_fail", withdraw(1000), [-1]))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
