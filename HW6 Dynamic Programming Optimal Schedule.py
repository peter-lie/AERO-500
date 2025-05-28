# Peter Lie
# AERO 500 / 470

# Homework 6: Dynamic Programming Optimal Schedule

import os
import numpy as np
clear = lambda: os.system('clear')
clear()

# Each event: (start, duration, value)
events = [
    (12, 4, 5),   # ends at 16
    (3, 4, 8),
    (7, 2, 8),
    (21, 3, 9),
    (23, 3, 5),
    (24, 1, 7),
    (1, 3, 5),
    (17, 2, 6),
    (13, 1, 6),
    (20, 4, 5)
]

# Convert to (index, start, end, value)
jobs = [(i, start, start + dur, value) for i, (start, dur, value) in enumerate(events)]

# Sort by end time
jobs.sort(key=lambda job: job[2])

# Binary search: find latest job j < i where job[j].end <= job[i].start
def latest_non_conflicting(jobs, i):
    low, high = 0, i - 1
    while low <= high:
        mid = (low + high) // 2
        if jobs[mid][2] <= jobs[i][1]:
            if jobs[mid + 1][2] <= jobs[i][1]:
                low = mid + 1
            else:
                return mid
        else:
            high = mid - 1
    return -1

# Initialize DP arrays
n = len(jobs)
dp = [0] * n
opt_choice = [-1] * n  # to reconstruct path

for i in range(n):
    incl = jobs[i][3]
    l = latest_non_conflicting(jobs, i)
    if l != -1:
        incl += dp[l]
    
    excl = dp[i - 1] if i > 0 else 0
    
    if incl > excl:
        dp[i] = incl
        opt_choice[i] = l
    else:
        dp[i] = excl
        opt_choice[i] = opt_choice[i - 1] if i > 0 else -1

# Reconstruct optimal schedule
def reconstruct_schedule(jobs, opt_choice):
    schedule = []
    i = len(jobs) - 1
    while i >= 0:
        prev = opt_choice[i]
        if prev == -1 and (i == 0 or dp[i] > dp[i - 1]):
            schedule.append(jobs[i][0])
            break
        elif prev != -1 and dp[i] != dp[i - 1]:
            schedule.append(jobs[i][0])
            i = prev
        else:
            i -= 1
    return sorted(schedule)


# Output
best_schedule = reconstruct_schedule(jobs, opt_choice)
# Sort by execution order (i.e., start time of each scheduled event)
best_schedule.sort(key=lambda idx: events[idx][0])

total_value = dp[-1]

print("Best Schedule:", best_schedule)
print("Total Value:", total_value)
print("Scheduled Events:")
for idx in best_schedule:
    s, d, v = events[idx]
    print(f"  Event {idx}: Start {s}, Duration {d}, End {s+d}, Value {v}")


