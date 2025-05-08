"""
Basic usage examples for the Weight-of-Thought model.

This example demonstrates how to load a pre-trained model
and use it for inference on different reasoning tasks.
"""

import torch
import matplotlib.pyplot as plt

from wot.models import WOTReasoner

# Set random seed for reproducibility
torch.manual_seed(42)

def main():
    # Load pre-trained model
    print("Loading pre-trained WOT model...")
    wot_reasoner = WOTReasoner()
    wot_reasoner.load_model('results/models/wot_model_final.pt')
    
    # Example reasoning tasks
    examples = [
        {
            'question': 'If all Bloops are Razzies and all Razzies are Wazzies, are all Bloops definitely Wazzies? Answer with Yes or No.',
            'type': 'syllogism'
        },
        {
            'question': 'What is the next number in the sequence: 2, 4, 6, 8, 10, ...?',
            'type': 'math_sequence'
        },
        {
            'question': 'John has 3 times as many apples as Mary. Together, they have 40 apples. How many apples does John have?',
            'type': 'algebra'
        },
        {
            'question': 'In a room of 15 people, everyone shakes hands with everyone else exactly once. How many handshakes are there in total?',
            'type': 'combinatorics'
        },
        {
            'question': 'Is every square a rectangle? Answer with Yes or No.',
            'type': 'geometry'
        }
    ]
    
    # Run inference on each example
    print("\nRunning inference on example tasks...")
    for example in examples:
        question = example['question']
        task_type = example['type']
        
        print(f"\nTask Type: {task_type}")
        print(f"Question: {question}")
        
        # Run inference
        answer = wot_reasoner.infer(question, task_type)
        print(f"Answer: {answer}")
        
        # Visualize attention (if the model has attention weights stored)
        if hasattr(wot_reasoner.wot_model, 'node_attention_weights') and wot_reasoner.wot_model.node_attention_weights is not None:
            print("Visualizing node attention weights...")
            node_attention = wot_reasoner.wot_model.node_attention_weights.numpy()
            
            # Create a simple visualization of node attention
            plt.figure(figsize=(8, 3))
            plt.bar(range(wot_reasoner.wot_model.num_nodes), node_attention.squeeze())
            plt.xlabel('Node Index')
            plt.ylabel('Attention Weight')
            plt.title(f'Node Attention for {task_type.capitalize()} Task')
            plt.tight_layout()
            plt.show()
    
    # Custom question
    print("\nTry your own question:")
    question = input("Enter your question: ")
    print("Select task type:")
    print("1. Syllogism")
    print("2. Math Sequence")
    print("3. Algebra")
    print("4. Combinatorics")
    print("5. Geometry")
    task_choice = input("Enter choice (1-5): ")
    
    task_types = ['syllogism', 'math_sequence', 'algebra', 'combinatorics', 'geometry']
    task_type = task_types[int(task_choice) - 1] if task_choice.isdigit() and 1 <= int(task_choice) <= 5 else 'syllogism'
    
    # Run inference on custom question
    answer = wot_reasoner.infer(question, task_type)
    print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()