#!/usr/bin/env python
"""
Evaluation script for Weight-of-Thought model.

This script evaluates a trained WoT model on various reasoning tasks
and generates performance metrics and visualizations.
"""

import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix
)

import torch
from torch.utils.data import DataLoader

from wot.models import WOTReasoner
from wot.data import ReasoningDataset
from wot.data.tasks import generate_all_tasks


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Weight-of-Thought Reasoning Model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    
    # Evaluation parameters
    parser.add_argument('--num_tasks', type=int, default=100,
                        help='Number of tasks to evaluate (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed performance analysis')
    
    # Output directories
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                        help='Directory to save evaluation results')
    
    return parser.parse_args()


def evaluate_model(model, dataloader):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: WOTReasoner model
        dataloader: DataLoader with evaluation data
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Prepare result containers
    class_predictions = []
    class_labels = []
    numeric_predictions = []
    numeric_labels = []
    
    task_type_results = {
        'syllogism': {'correct': 0, 'total': 0},
        'math_sequence': {'predictions': [], 'labels': []},
        'algebra': {'predictions': [], 'labels': []},
        'combinatorics': {'predictions': [], 'labels': []},
        'geometry': {'correct': 0, 'total': 0}
    }
    
    # Track inference time
    inference_times = []
    
    # Run evaluation
    model.encoder.eval()
    model.wot_model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            # Group by task type for efficient processing
            task_types = batch['task_type']
            unique_task_types = set(task_types)
            
            for task_type in unique_task_types:
                # Get indices for this task type
                indices = [i for i, t in enumerate(task_types) if t == task_type]
                
                if not indices:
                    continue
                
                # Get batch subset for this task type
                batch_input_ids = input_ids[indices]
                batch_attention_mask = attention_mask[indices]
                
                # Time the inference
                start_time = time.time()
                
                # Encode the input text
                text_embedding = model.encoder(batch_input_ids, batch_attention_mask)
                
                # Forward pass through WOT model
                class_logits, numeric_prediction = model.wot_model(text_embedding, task_type)
                
                # Record inference time
                inference_times.append((time.time() - start_time) / len(indices))
                
                if task_type in ['syllogism', 'geometry']:
                    # Classification task
                    labels = batch['label'][indices].to(model.device)
                    _, predicted = torch.max(class_logits, 1)
                    
                    # Add to overall results
                    class_predictions.extend(predicted.cpu().numpy())
                    class_labels.extend(labels.cpu().numpy())
                    
                    # Add to task-specific results
                    task_type_results[task_type]['correct'] += (predicted == labels).sum().item()
                    task_type_results[task_type]['total'] += labels.size(0)
                else:
                    # Regression task
                    labels = batch['numeric_label'][indices].to(model.device).float()
                    
                    # Ensure prediction shape matches labels
                    predictions = numeric_prediction.view(-1)
                    labels = labels.view(-1)
                    
                    # Add to overall results
                    numeric_predictions.extend(predictions.cpu().numpy())
                    numeric_labels.extend(labels.cpu().numpy())
                    
                    # Add to task-specific results
                    task_type_results[task_type]['predictions'].extend(predictions.cpu().numpy())
                    task_type_results[task_type]['labels'].extend(labels.cpu().numpy())
    
    # Calculate overall metrics
    results = {
        'inference_time': np.mean(inference_times)
    }
    
    # Classification metrics
    if class_labels:
        results['classification'] = {
            'accuracy': accuracy_score(class_labels, class_predictions),
            'precision_recall_f1': precision_recall_fscore_support(
                class_labels, class_predictions, average='weighted'
            )
        }
    
    # Regression metrics
    if numeric_labels:
        results['regression'] = {
            'mse': mean_squared_error(numeric_labels, numeric_predictions),
            'mae': mean_absolute_error(numeric_labels, numeric_predictions),
            'r2': r2_score(numeric_labels, numeric_predictions)
        }
    
    # Task-specific metrics
    for task_type, task_results in task_type_results.items():
        if task_type in ['syllogism', 'geometry'] and task_results['total'] > 0:
            results[task_type] = {
                'accuracy': task_results['correct'] / task_results['total']
            }
        elif (task_type in ['math_sequence', 'algebra', 'combinatorics'] and 
              len(task_results['labels']) > 0):
            results[task_type] = {
                'mse': mean_squared_error(task_results['labels'], task_results['predictions']),
                'mae': mean_absolute_error(task_results['labels'], task_results['predictions'])
            }
    
    return results


def generate_visualizations(evaluation_results, output_dir):
    """
    Generate visualizations of evaluation results.
    
    Args:
        evaluation_results: Dictionary of evaluation metrics
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Task-specific performance bar chart
    plt.figure(figsize=(12, 6))
    
    # Classification performance
    classification_tasks = ['syllogism', 'geometry']
    classification_scores = [evaluation_results.get(task, {}).get('accuracy', 0) 
                            for task in classification_tasks]
    
    # Regression performance (inverted MSE, so higher is better)
    regression_tasks = ['math_sequence', 'algebra', 'combinatorics']
    regression_scores = [1.0 / (evaluation_results.get(task, {}).get('mse', float('inf')) + 0.1) 
                         for task in regression_tasks]
    
    # All task names and scores
    all_tasks = classification_tasks + regression_tasks
    all_scores = classification_scores + regression_scores
    
    # Create a colormap for the bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_tasks)))
    
    # Create bar chart
    bars = plt.bar(all_tasks, all_scores, color=colors)
    
    # Add score labels on top of the bars
    for bar, task in zip(bars, all_tasks):
        if task in classification_tasks:
            score = evaluation_results.get(task, {}).get('accuracy', 0)
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{score:.2f}', ha='center', va='bottom')
        else:
            mse = evaluation_results.get(task, {}).get('mse', float('inf'))
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'MSE: {mse:.2f}', ha='center', va='bottom')
    
    plt.ylabel('Performance Score')
    plt.title('Task-Specific Performance')
    plt.ylim(0, 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task_performance.png'))
    plt.close()
    
    # 2. Inference time comparison
    plt.figure(figsize=(10, 5))
    
    # Create a bar for each task type
    inference_time = evaluation_results['inference_time']
    plt.bar(['Weight-of-Thought'], [inference_time], color='skyblue')
    plt.text(0, inference_time + 0.002, f'{inference_time:.3f}s', 
             ha='center', va='bottom')
    
    plt.ylabel('Average Inference Time (seconds)')
    plt.title('Model Inference Time')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_time.png'))
    plt.close()
    
    # Save numerical results as CSV
    results_df = pd.DataFrame({
        'Task': all_tasks + ['Average'],
        'Performance': all_scores + [np.mean(all_scores)],
        'Metric': ['Accuracy' if task in classification_tasks else 'Inverse MSE' 
                  for task in all_tasks] + ['Mixed']
    })
    results_df.to_csv(os.path.join(output_dir, 'performance_results.csv'), index=False)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Initializing WOT Reasoner...")
    wot_reasoner = WOTReasoner()
    
    print(f"Loading model from {args.model_path}")
    wot_reasoner.load_model(args.model_path)
    
    # Generate evaluation tasks
    print(f"Generating {args.num_tasks} evaluation tasks...")
    tasks_per_type = args.num_tasks // 5  # 5 task types
    eval_tasks = generate_all_tasks(num_each=tasks_per_type)
    
    # Create dataset and dataloader
    print("Preparing evaluation dataset...")
    eval_dataset = ReasoningDataset(eval_tasks, wot_reasoner.encoder.tokenizer)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Run evaluation
    print("\nEvaluating model...")
    start_time = time.time()
    evaluation_results = evaluate_model(wot_reasoner, eval_loader)
    evaluation_time = time.time() - start_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Inference Time: {evaluation_results['inference_time']:.4f} seconds per sample")
    
    if 'classification' in evaluation_results:
        print("\nClassification Tasks:")
        print(f"Overall Accuracy: {evaluation_results['classification']['accuracy']:.4f}")
        precision, recall, f1, _ = evaluation_results['classification']['precision_recall_f1']
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    if 'regression' in evaluation_results:
        print("\nRegression Tasks:")
        print(f"MSE: {evaluation_results['regression']['mse']:.4f}")
        print(f"MAE: {evaluation_results['regression']['mae']:.4f}")
        print(f"R²: {evaluation_results['regression']['r2']:.4f}")
    
    # Task-specific results
    print("\nTask-Specific Results:")
    for task_type in ['syllogism', 'math_sequence', 'algebra', 'combinatorics', 'geometry']:
        if task_type in evaluation_results:
            print(f"\n{task_type.capitalize()}:")
            for metric, value in evaluation_results[task_type].items():
                print(f"  {metric}: {value:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(evaluation_results, args.output_dir)
    
    # Save full results as JSON
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj
        
        json.dump(convert_numpy(evaluation_results), f, indent=2)
    
    print(f"\nEvaluation results saved to {args.output_dir}")


if __name__ == "__main__":
    main()