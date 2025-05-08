#!/usr/bin/env python
"""
Training script for Weight-of-Thought model.
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from wot.models import WOTReasoner
from wot.data import ReasoningDataset
from wot.data.tasks import tasks  # This will be the imported tasks from tasks.py


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Weight-of-Thought Reasoning Model')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=256, 
                        help='Hidden dimension size')
    parser.add_argument('--num_nodes', type=int, default=8, 
                        help='Number of nodes in the WOT network')
    parser.add_argument('--num_reasoning_steps', type=int, default=4, 
                        help='Number of reasoning steps')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-5, 
                        help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='Proportion of data to use for testing')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    # Model loading/saving
    parser.add_argument('--load_model', type=str, default=None, 
                        help='Path to load a pre-trained model')
    parser.add_argument('--save_dir', type=str, default='results/models', 
                        help='Directory to save models')
    parser.add_argument('--save_interval', type=int, default=5, 
                        help='Save model every N epochs')
    
    # Execution modes
    parser.add_argument('--inference_only', action='store_true', 
                        help='Run inference only, no training')
    parser.add_argument('--compare', action='store_true', 
                        help='Compare with other reasoning methods')
    parser.add_argument('--fast', action='store_true', 
                        help='Use reduced parameters for faster execution (testing)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Adjust parameters if in fast mode (for testing)
    if args.fast:
        print("Using fast training mode with reduced parameters")
        args.epochs = 3
        args.batch_size = 4
        args.hidden_dim = 64
        args.num_nodes = 3
        args.num_reasoning_steps = 2
    
    # Create directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Load and prepare data
    all_tasks = tasks
    train_tasks, test_tasks = train_test_split(
        all_tasks, test_size=args.test_size, random_state=args.seed
    )
    
    print(f"Training on {len(train_tasks)} tasks, testing on {len(test_tasks)} tasks")
    
    # Initialize model
    print("Initializing WOT Reasoner")
    wot_reasoner = WOTReasoner(
        hidden_dim=args.hidden_dim,
        num_nodes=args.num_nodes,
        num_reasoning_steps=args.num_reasoning_steps,
        lr=args.lr
    )
    
    # Create datasets and dataloaders
    print("Preparing datasets")
    encoder = wot_reasoner.encoder  # Get the encoder from the model
    train_dataset = ReasoningDataset(train_tasks, encoder.tokenizer)
    test_dataset = ReasoningDataset(test_tasks, encoder.tokenizer)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, drop_last=False
    )
    
    # Load pre-trained model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        wot_reasoner.load_model(args.load_model)
    
    # Print model configuration
    print("\nModel configuration:")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of nodes: {args.num_nodes}")
    print(f"  Reasoning steps: {args.num_reasoning_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    
    # Training phase
    if not args.inference_only:
        print("\nStarting training...")
        start_time = time.time()
        history = wot_reasoner.train(
            train_loader, test_loader, num_epochs=args.epochs
        )
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")
        
        # Save final model
        final_model_path = os.path.join(args.save_dir, 'wot_model_final.pt')
        print(f"Saving final model to {final_model_path}")
        wot_reasoner.save_model(final_model_path)
    
    # Evaluation phase
    print("\nEvaluating on test set...")
    test_loss, test_class_acc, test_mse = wot_reasoner.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Class Acc: {test_class_acc:.4f}, MSE: {test_mse:.4f}")
    
    # Run inference examples
    print("\nRunning inference examples...")
    for task_type in ['syllogism', 'math_sequence', 'algebra', 'combinatorics', 'geometry']:
        # Find a task of this type in the test set
        for task in test_tasks:
            if task['type'] == task_type:
                question = task['question']
                true_answer = task['answer']
                
                # Run inference
                start_time = time.time()
                predicted_answer = wot_reasoner.infer(question, task_type)
                inference_time = time.time() - start_time
                
                print(f"\nTask type: {task_type}")
                print(f"Question: {question}")
                print(f"True answer: {true_answer}")
                print(f"Predicted answer: {predicted_answer}")
                print(f"Inference time: {inference_time:.4f} seconds")
                print(f"Correct: {str(true_answer) == predicted_answer}")
                break
    
    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()