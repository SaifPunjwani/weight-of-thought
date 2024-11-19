import random

tasks = []

# Syllogism tasks (4 tasks)
for i in range(4):
    animals = ['Bloops', 'Razzies', 'Lazzies', 'Wazzies', 'Flazzies', 'Dazzies', 'Tazzies', 'Mazzies', 'Pazzies', 'Kazzies',
               'Quazzies', 'Yazzies', 'Nazzies', 'Hazzies', 'Gazzies', 'Jazzies', 'Xazzies', 'Vazzies', 'Bazzies', 'Zazzies']
    a1, a2, a3 = random.sample(animals, 3)
    tasks.append({
        'task_id': len(tasks) + 1,
        'question': f'If all {a1} are {a2} and all {a2} are {a3}, are all {a1} definitely {a3}? Answer with Yes or No.',
        'answer': 'Yes',
        'type': 'syllogism'
    })

# Math sequence tasks (4 tasks)
sequences = [
    ([2,4,6,8,10], '12', 'arithmetic +2'),
    ([2,4,8,16,32], '64', 'geometric *2'),
    ([1,1,2,3,5,8], '13', 'fibonacci'),
    ([1,4,9,16,25], '36', 'square numbers'),
    ([1,3,6,10,15], '21', 'triangular numbers'),
    ([3,6,9,12,15], '18', 'arithmetic +3'),
    ([1,3,9,27,81], '243', 'geometric *3'),
    ([2,6,12,20,30], '42', 'quadratic n(n+1)'),
    ([1,8,27,64,125], '216', 'cubic numbers'),
    ([1,2,4,7,11], '16', 'fibonacci variant')
]
for i in range(4):
    seq, next_num, seq_type = random.choice(sequences)
    tasks.append({
        'task_id': len(tasks) + 1,
        'question': f'What is the next number in the sequence: {", ".join(str(x) for x in seq)}, ...?',
        'answer': next_num,
        'type': 'math_sequence'
    })

# Algebra tasks (4 tasks)
for i in range(4):
    names = ['John', 'Mary', 'Bob', 'Alice', 'Tom', 'Sarah', 'Mike', 'Emma', 'David', 'Lisa',
             'James', 'Emily', 'William', 'Sophia', 'Oliver', 'Isabella', 'Henry', 'Mia', 'Alexander', 'Charlotte']
    n1, n2 = random.sample(names, 2)
    total = random.randint(10, 100)
    ratio = random.randint(2, 8)
    p1 = (ratio * total) // (ratio + 1)
    tasks.append({
        'task_id': len(tasks) + 1,
        'question': f'{n1} has {ratio} times as many apples as {n2}. Together, they have {total} apples. How many apples does {n1} have?',
        'answer': str(p1),
        'type': 'algebra'
    })

# Combinatorics tasks (4 tasks)
for i in range(4):
    n = random.randint(5, 30)
    handshakes = (n * (n-1)) // 2
    tasks.append({
        'task_id': len(tasks) + 1,
        'question': f'In a room of {n} people, everyone shakes hands with everyone else exactly once. How many handshakes are there in total?',
        'answer': str(handshakes),
        'type': 'combinatorics'
    })

# Geometry tasks (4 tasks)
geometry_questions = [
    ('Is it possible for a square to have more than four sides?', 'No'),
    ('Can a triangle have two right angles?', 'No'),
    ('Is every square a rectangle?', 'Yes'),
    ('Is every rectangle a square?', 'No'),
    ('Can a circle have corners?', 'No'),
    ('Can a triangle have three equal angles?', 'Yes'),
    ('Is it possible for a rectangle to have all sides equal?', 'Yes'),
    ('Can a pentagon have four right angles?', 'No'),
    ('Does every parallelogram have four equal sides?', 'No'),
    ('Can a trapezoid have all sides equal?', 'No')
]
for i in range(4):
    q, a = random.choice(geometry_questions)
    tasks.append({
        'task_id': len(tasks) + 1,
        'question': q + ' Answer with Yes or No.',
        'answer': a,
        'type': 'geometry'
    })