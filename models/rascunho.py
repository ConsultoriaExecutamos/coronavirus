n = 34

step_size = 3

number_of_steps = n // 3 + 1
last_step_size = n % 3

print(number_of_steps, last_step_size)

schedule = []

for i in range(number_of_steps - 1):
    schedule.append(step_size)

schedule.append(last_step_size)

print(schedule)