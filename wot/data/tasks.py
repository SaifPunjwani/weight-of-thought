"""
Predefined reasoning tasks for the Weight-of-Thought model.

This module contains a collection of reasoning tasks across different categories:
- Syllogism: Logical deduction tasks
- Math Sequence: Pattern recognition in number sequences
- Algebra: Solving algebraic word problems
- Combinatorics: Counting and arrangement problems
- Geometry: Geometric reasoning tasks

These tasks can be used for training and evaluating the Weight-of-Thought model.
"""

import random


def generate_syllogism_tasks(num_tasks=10):
    """
    Generate syllogistic reasoning tasks.
    
    Args:
        num_tasks: Number of tasks to generate (default: 10)
        
    Returns:
        List of syllogism tasks
    """
    tasks = []
    animals = [
        'Bloops', 'Razzies', 'Lazzies', 'Wazzies', 'Flazzies', 
        'Dazzies', 'Tazzies', 'Mazzies', 'Pazzies', 'Kazzies',
        'Quazzies', 'Yazzies', 'Nazzies', 'Hazzies', 'Gazzies', 
        'Jazzies', 'Xazzies', 'Vazzies', 'Bazzies', 'Zazzies'
    ]
    
    # Valid syllogisms (All A are B, All B are C, therefore All A are C)
    for i in range(num_tasks // 2):
        a1, a2, a3 = random.sample(animals, 3)
        tasks.append({
            'task_id': len(tasks) + 1,
            'question': f'If all {a1} are {a2} and all {a2} are {a3}, are all {a1} definitely {a3}? Answer with Yes or No.',
            'answer': 'Yes',
            'type': 'syllogism'
        })
    
    # Invalid syllogisms (All A are B, Some B are C, therefore All A are C - invalid)
    for i in range(num_tasks - len(tasks)):
        a1, a2, a3 = random.sample(animals, 3)
        tasks.append({
            'task_id': len(tasks) + 1,
            'question': f'If all {a1} are {a2} and some {a2} are {a3}, are all {a1} definitely {a3}? Answer with Yes or No.',
            'answer': 'No',
            'type': 'syllogism'
        })
    
    return tasks


def generate_math_sequence_tasks(num_tasks=10):
    """
    Generate mathematical sequence tasks.
    
    Args:
        num_tasks: Number of tasks to generate (default: 10)
        
    Returns:
        List of math sequence tasks
    """
    tasks = []
    sequences = [
        ([2, 4, 6, 8, 10], '12', 'arithmetic +2'),
        ([2, 4, 8, 16, 32], '64', 'geometric *2'),
        ([1, 1, 2, 3, 5, 8], '13', 'fibonacci'),
        ([1, 4, 9, 16, 25], '36', 'square numbers'),
        ([1, 3, 6, 10, 15], '21', 'triangular numbers'),
        ([3, 6, 9, 12, 15], '18', 'arithmetic +3'),
        ([1, 3, 9, 27, 81], '243', 'geometric *3'),
        ([2, 6, 12, 20, 30], '42', 'quadratic n(n+1)'),
        ([1, 8, 27, 64, 125], '216', 'cubic numbers'),
        ([1, 2, 4, 7, 11], '16', 'fibonacci variant')
    ]
    
    for i in range(num_tasks):
        seq, next_num, seq_type = random.choice(sequences)
        tasks.append({
            'task_id': len(tasks) + 1,
            'question': f'What is the next number in the sequence: {", ".join(str(x) for x in seq)}, ...?',
            'answer': next_num,
            'type': 'math_sequence'
        })
    
    return tasks


def generate_algebra_tasks(num_tasks=10):
    """
    Generate algebraic word problems.
    
    Args:
        num_tasks: Number of tasks to generate (default: 10)
        
    Returns:
        List of algebra tasks
    """
    tasks = []
    names = [
        'John', 'Mary', 'Bob', 'Alice', 'Tom', 'Sarah', 'Mike', 'Emma', 
        'David', 'Lisa', 'James', 'Emily', 'William', 'Sophia', 'Oliver', 
        'Isabella', 'Henry', 'Mia', 'Alexander', 'Charlotte'
    ]
    
    for i in range(num_tasks):
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
    
    return tasks


def generate_combinatorics_tasks(num_tasks=10):
    """
    Generate combinatorial reasoning tasks.
    
    Args:
        num_tasks: Number of tasks to generate (default: 10)
        
    Returns:
        List of combinatorics tasks
    """
    tasks = []
    
    for i in range(num_tasks):
        n = random.randint(5, 30)
        handshakes = (n * (n-1)) // 2
        tasks.append({
            'task_id': len(tasks) + 1,
            'question': f'In a room of {n} people, everyone shakes hands with everyone else exactly once. How many handshakes are there in total?',
            'answer': str(handshakes),
            'type': 'combinatorics'
        })
    
    return tasks


def generate_geometry_tasks(num_tasks=10):
    """
    Generate geometric reasoning tasks.
    
    Args:
        num_tasks: Number of tasks to generate (default: 10)
        
    Returns:
        List of geometry tasks
    """
    tasks = []
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
    
    for i in range(num_tasks):
        q, a = random.choice(geometry_questions)
        tasks.append({
            'task_id': len(tasks) + 1,
            'question': q + ' Answer with Yes or No.',
            'answer': a,
            'type': 'geometry'
        })
    
    return tasks


def generate_all_tasks(num_each=5):
    """
    Generate a mixed set of tasks across all reasoning types.
    
    Args:
        num_each: Number of tasks to generate for each type (default: 5)
        
    Returns:
        List of all reasoning tasks
    """
    all_tasks = []
    all_tasks.extend(generate_syllogism_tasks(num_each))
    all_tasks.extend(generate_math_sequence_tasks(num_each))
    all_tasks.extend(generate_algebra_tasks(num_each))
    all_tasks.extend(generate_combinatorics_tasks(num_each))
    all_tasks.extend(generate_geometry_tasks(num_each))
    
    # Shuffle tasks
    random.shuffle(all_tasks)
    
    return all_tasks


# Generate a default set of tasks
tasks = generate_all_tasks(num_each=5)


if __name__ == "__main__":
    # Example usage
    print(f"Generated {len(tasks)} reasoning tasks:")
    
    # Print one example of each type
    task_types = ['syllogism', 'math_sequence', 'algebra', 'combinatorics', 'geometry']
    for task_type in task_types:
        for task in tasks:
            if task['type'] == task_type:
                print(f"\nTask Type: {task_type}")
                print(f"Question: {task['question']}")
                print(f"Answer: {task['answer']}")
                break