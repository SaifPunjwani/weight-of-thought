"""
Example demonstrating how to create and use custom reasoning tasks with the WoT model.

This example shows how to:
1. Define custom reasoning tasks
2. Train the WoT model on these tasks
3. Evaluate the model on the custom tasks
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from wot.models import WOTReasoner
from wot.data import ReasoningDataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define custom reasoning tasks
def create_custom_tasks():
    """
    Create custom reasoning tasks for training and evaluation.
    
    In this example, we'll create:
    1. Logical equivalence tasks (classification)
    2. Function composition tasks (regression)
    
    Returns:
        List of task dictionaries
    """
    custom_tasks = []
    
    # 1. Logical equivalence tasks
    # Determine if two logical statements are equivalent
    equivalence_pairs = [
        # Equivalent statements (commutative, associative, distributive, etc.)
        ("p AND q", "q AND p", "Yes"),
        ("p OR q", "q OR p", "Yes"),
        ("NOT (p AND q)", "NOT p OR NOT q", "Yes"),  # De Morgan's law
        ("NOT (p OR q)", "NOT p AND NOT q", "Yes"),  # De Morgan's law
        ("p AND (q OR r)", "(p AND q) OR (p AND r)", "Yes"),  # Distributive property
        ("p OR (q AND r)", "(p OR q) AND (p OR r)", "Yes"),  # Distributive property
        
        # Non-equivalent statements
        ("p AND q", "p OR q", "No"),
        ("p AND NOT p", "q AND NOT q", "No"),
        ("p OR NOT q", "NOT p OR q", "No"),
        ("p AND (q OR r)", "p OR (q AND r)", "No"),
        ("(p AND q) OR r", "p AND (q OR r)", "No"),
        ("NOT (p AND q)", "NOT p AND NOT q", "No")
    ]
    
    for i, (stmt1, stmt2, answer) in enumerate(equivalence_pairs):
        custom_tasks.append({
            'task_id': len(custom_tasks) + 1,
            'question': f'Are the logical statements "{stmt1}" and "{stmt2}" equivalent? Answer with Yes or No.',
            'answer': answer,
            'type': 'logical_equivalence'
        })
    
    # 2. Function composition tasks
    # Calculate the result of function composition
    compositions = [
        # Linear functions
        ("f(x) = 2x + 3, g(x) = x - 1", "f(g(5))", "8"),
        ("f(x) = 3x - 1, g(x) = x + 2", "f(g(2))", "11"),
        ("f(x) = x/2, g(x) = 4x", "f(g(3))", "6"),
        
        # Polynomial functions
        ("f(x) = x^2, g(x) = x + 1", "f(g(2))", "9"),
        ("f(x) = x^2 + 1, g(x) = 2x", "f(g(3))", "37"),
        ("f(x) = 2x^2 - 1, g(x) = x + 2", "f(g(1))", "17"),
        
        # Mixed functions
        ("f(x) = 2x + 1, g(x) = x^2", "f(g(3))", "19"),
        ("f(x) = x^2, g(x) = 2x + 1", "f(g(2))", "25"),
        ("f(x) = |x|, g(x) = x - 5", "f(g(-2))", "7"),
        ("f(x) = x^3, g(x) = x/2", "f(g(4))", "8")
    ]
    
    for i, (functions, expr, result) in enumerate(compositions):
        custom_tasks.append({
            'task_id': len(custom_tasks) + 1,
            'question': f'Given {functions}, calculate {expr}.',
            'answer': result,
            'type': 'function_composition'
        })
    
    return custom_tasks


def main():
    # Create custom tasks
    custom_tasks = create_custom_tasks()
    print(f"Created {len(custom_tasks)} custom reasoning tasks")
    
    # Split into training and testing sets
    train_tasks, test_tasks = train_test_split(custom_tasks, test_size=0.3, random_state=42)
    print(f"Training tasks: {len(train_tasks)}, Testing tasks: {len(test_tasks)}")
    
    # Initialize the WoT model
    print("Initializing WoT Reasoner...")
    wot_reasoner = WOTReasoner(
        hidden_dim=128,
        num_nodes=6,
        num_reasoning_steps=3,
        lr=3e-5
    )
    
    # Extend the model's output types to handle our custom task types
    print("Extending model to handle custom task types...")
    
    # Needed when using an instance directly, although typically this would be modified in the class
    wot_reasoner.wot_model.output_types['logical_equivalence'] = wot_reasoner.wot_model.output_types['syllogism']
    wot_reasoner.wot_model.output_types['function_composition'] = wot_reasoner.wot_model.output_types['math_sequence']
    
    # Create datasets
    print("Creating datasets...")
    encoder = wot_reasoner.encoder
    train_dataset = ReasoningDataset(train_tasks, encoder.tokenizer)
    test_dataset = ReasoningDataset(test_tasks, encoder.tokenizer)
    
    # Create dataloaders
    batch_size = 4  # Small batch size for this example
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Train the model
    print("Training model on custom tasks...")
    train_option = input("Do you want to train the model? (y/n): ")
    
    if train_option.lower() == 'y':
        history = wot_reasoner.train(train_loader, test_loader, num_epochs=5)
        print("Training completed!")
        
        # Visualize training results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Test on some examples
    print("\nTesting on custom examples:")
    
    # Choose some examples from the test set
    example_indices = [0, len(test_tasks) // 3, 2 * len(test_tasks) // 3]
    for idx in example_indices:
        task = test_tasks[idx]
        question = task['question']
        true_answer = task['answer']
        task_type = task['type']
        
        print(f"\nTask Type: {task_type}")
        print(f"Question: {question}")
        print(f"True Answer: {true_answer}")
        
        # Make prediction
        predicted_answer = wot_reasoner.infer(question, task_type)
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Correct: {str(true_answer) == predicted_answer}")
    
    # Create your own custom question
    print("\nTry your own custom question:")
    print("1. Logical equivalence")
    print("2. Function composition")
    task_choice = input("Enter choice (1-2): ")
    
    if task_choice == '1':
        stmt1 = input("Enter first logical statement: ")
        stmt2 = input("Enter second logical statement: ")
        question = f'Are the logical statements "{stmt1}" and "{stmt2}" equivalent? Answer with Yes or No.'
        task_type = 'logical_equivalence'
    else:
        functions = input("Enter functions (e.g., f(x) = 2x + 1, g(x) = x^2): ")
        expr = input("Enter expression to evaluate (e.g., f(g(3))): ")
        question = f'Given {functions}, calculate {expr}.'
        task_type = 'function_composition'
    
    # Make prediction
    print("\nQuestion:", question)
    predicted_answer = wot_reasoner.infer(question, task_type)
    print(f"Predicted Answer: {predicted_answer}")


if __name__ == "__main__":
    main()