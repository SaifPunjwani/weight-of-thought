#!/usr/bin/env python
"""
Weight-of-Thought Quick Start Example

This script provides a quick demonstration of the Weight-of-Thought (WoT)
reasoning model. It loads a pre-trained model and runs inference on
several example reasoning tasks.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path to ensure we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wot.models import WOTReasoner

def main():
    """Run a quick demonstration of the WoT model."""
    print("Weight-of-Thought (WoT) Demo")
    print("============================")
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize the WoT reasoner
    print("\nInitializing WoT Reasoner...")
    reasoner = WOTReasoner()
    
    # Check if pre-trained model exists
    model_path = 'results/models/wot_model_final.pt'
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        reasoner.load_model(model_path)
    else:
        print("No pre-trained model found. Using a newly initialized model.")
        print("Note: For better results, train the model before inference.")
    
    # Example reasoning tasks
    example_tasks = [
        {
            "question": "If all Bloops are Razzies and all Razzies are Wazzies, are all Bloops definitely Wazzies? Answer with Yes or No.",
            "type": "syllogism",
            "description": "Logical syllogism (transitive property)"
        },
        {
            "question": "What is the next number in the sequence: 2, 4, 6, 8, 10, ...?",
            "type": "math_sequence",
            "description": "Mathematical sequence prediction"
        },
        {
            "question": "John has 3 times as many apples as Mary. Together, they have 40 apples. How many apples does John have?",
            "type": "algebra",
            "description": "Algebraic word problem"
        },
        {
            "question": "In a room of 10 people, everyone shakes hands with everyone else exactly once. How many handshakes are there in total?",
            "type": "combinatorics",
            "description": "Combinatorial counting problem"
        },
        {
            "question": "Is every square a rectangle? Answer with Yes or No.",
            "type": "geometry",
            "description": "Geometric reasoning"
        }
    ]
    
    # Run inference on each example
    print("\nRunning inference on example tasks...")
    
    for task in example_tasks:
        print(f"\n--- {task['description']} ---")
        print(f"Question: {task['question']}")
        
        # Run inference
        start_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
        
        import time
        if start_time:
            start_time.record()
        else:
            start = time.time()
            
        answer = reasoner.infer(task['question'], task['type'])
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            inference_time = time.time() - start
        
        print(f"Answer: {answer}")
        print(f"Inference time: {inference_time:.4f} seconds")
    
    # Interactive mode
    try_yourself = input("\nWould you like to try your own question? (y/n): ")
    
    if try_yourself.lower() == 'y':
        print("\nSelect a reasoning task type:")
        print("1. Syllogism (logical deduction)")
        print("2. Math Sequence (pattern recognition)")
        print("3. Algebra (word problems)")
        print("4. Combinatorics (counting problems)")
        print("5. Geometry (geometric properties)")
        
        choice = input("Enter your choice (1-5): ")
        try:
            choice_num = int(choice)
            task_types = ["syllogism", "math_sequence", "algebra", "combinatorics", "geometry"]
            task_type = task_types[choice_num - 1] if 1 <= choice_num <= 5 else "syllogism"
            
            question = input("\nEnter your question: ")
            
            print("\nRunning inference...")
            answer = reasoner.infer(question, task_type)
            print(f"Answer: {answer}")
            
        except (ValueError, IndexError):
            print("Invalid choice. Using syllogism as default.")
            question = input("\nEnter your question: ")
            answer = reasoner.infer(question, "syllogism")
            print(f"Answer: {answer}")
    
    print("\nDemo completed. Explore more examples in the examples directory.")
    print("For detailed documentation, refer to the README.md file.")

if __name__ == "__main__":
    main()