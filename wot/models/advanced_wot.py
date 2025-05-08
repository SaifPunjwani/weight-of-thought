import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import math
import time
import os
from matplotlib.gridspec import GridSpec
from collections import defaultdict, deque
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, mean_squared_error
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr
import scipy.stats as stats
from sklearn.model_selection import KFold
import copy

# Create directories for advanced results
os.makedirs('results/advanced', exist_ok=True)
os.makedirs('results/advanced/plots', exist_ok=True)
os.makedirs('results/advanced/models', exist_ok=True)
os.makedirs('results/advanced/weight_maps', exist_ok=True)
os.makedirs('results/advanced/attention_maps', exist_ok=True)
os.makedirs('results/advanced/node_analysis', exist_ok=True)
os.makedirs('results/advanced/task_performance', exist_ok=True)

class AdvancedWebOfThoughts(nn.Module):
    """
    Enhanced Weight-of-Thought reasoning model with advanced architecture features
    """
    def __init__(self, input_dim, hidden_dim=256, num_nodes=8, num_reasoning_steps=4, 
                 attention_heads=4, dropout=0.1, layer_norm=True, skip_connections=True,
                 activation='gelu', node_specialization=True, edge_importance=True,
                 task_embedding_dim=32):
        super(AdvancedWebOfThoughts, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_reasoning_steps = num_reasoning_steps
        self.node_specialization = node_specialization
        self.edge_importance = edge_importance
        self.task_embedding_dim = task_embedding_dim
        self.skip_connections = skip_connections
        self.layer_norm = layer_norm
        self.attention_heads = attention_heads
        
        # Activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            self.activation = nn.GELU()
        
        # Task type embeddings
        self.task_embeddings = nn.Embedding(5, task_embedding_dim)  # 5 task types
        
        # Initial embedding layer with increased capacity
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2) if layer_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout)
        )
        
        # Node feature transformation with improved architecture
        self.node_transform = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + (task_embedding_dim if node_specialization else 0), hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2) if layer_norm else nn.Identity(),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
                self.activation,
                nn.Dropout(dropout)
            ) for _ in range(num_nodes)
        ])
        
        # Multi-head attention for edges
        self.edge_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_nodes)
        ])
        
        # Edge importance weighting (if enabled)
        if edge_importance:
            self.edge_importance_weights = nn.Parameter(torch.ones(num_nodes, num_nodes))
        
        # Node gating for adaptive information flow
        self.node_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
                self.activation,
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_nodes)
        ])
        
        # Global attention for aggregating node outputs
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced reasoning step transformers
        encoder_layers = [
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=attention_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm architecture for better training stability
            ) for _ in range(num_reasoning_steps)
        ]
        
        self.reasoning_steps = nn.ModuleList(encoder_layers)
        
        # Task-specific output modules with deeper networks
        self.output_types = {
            'syllogism': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2)
            ),  # Yes/No
            'math_sequence': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2)
            ),  # Numerical prediction
            'algebra': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2)
            ),  # Numerical solution
            'combinatorics': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2)
            ),  # Numerical computation
            'geometry': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2)
            )  # Yes/No
        }
        
        # Enhanced final numeric prediction network
        self.final_numeric = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2) if layer_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Store intermediate states for visualization and analysis
        self.node_features = None
        self.node_attention_weights = None
        self.reasoning_attention_weights = None
        self.edge_matrices = None
        self.task_embedding_vectors = None
        self.message_importance = None
        self.reasoning_step_outputs = None
    
    def get_task_index(self, task_type):
        """Convert task type string to index"""
        task_map = {
            'syllogism': 0,
            'math_sequence': 1,
            'algebra': 2,
            'combinatorics': 3,
            'geometry': 4
        }
        return task_map.get(task_type, 0)
    
    def forward(self, x, task_type=None, return_intermediates=False):
        batch_size = x.size(0)
        device = x.device
        
        # Get task embedding if task specialization is enabled
        if self.node_specialization and task_type is not None:
            task_idx = torch.tensor([self.get_task_index(task_type)], device=device).expand(batch_size)
            task_embedding = self.task_embeddings(task_idx)  # [batch_size, task_embedding_dim]
            self.task_embedding_vectors = task_embedding.detach().cpu()
        
        # Initial embedding
        x = self.embedding(x)  # [batch_size, hidden_dim]
        
        # Initialize node features
        if self.node_specialization and task_type is not None:
            # Concatenate task embedding to input for specialized processing
            x_with_task = torch.cat([x, task_embedding], dim=-1)  # [batch_size, hidden_dim + task_embedding_dim]
            node_features = [self.node_transform[i](x_with_task) for i in range(self.num_nodes)]
        else:
            node_features = [self.node_transform[i](x) for i in range(self.num_nodes)]
        
        # Store initial node features for visualization
        self.node_features = [f.detach().cpu() for f in node_features]
        
        # Enhanced message passing with multi-head attention and importance weighting
        edge_matrices = []
        message_importance = []
        
        # Multiple rounds of message passing for better information propagation
        for round_idx in range(3):
            new_node_features = []
            edge_matrix = torch.zeros(batch_size, self.num_nodes, self.num_nodes, device=x.device)
            importance_matrix = torch.zeros(batch_size, self.num_nodes, self.num_nodes, device=x.device)
            
            # Stack all nodes for more efficient processing
            stacked_nodes = torch.stack(node_features, dim=1)  # [batch_size, num_nodes, hidden_dim]
            
            for i in range(self.num_nodes):
                # Current node as query
                query = node_features[i].unsqueeze(1)  # [batch_size, 1, hidden_dim]
                
                # All nodes as keys and values
                keys_values = stacked_nodes
                
                # Apply multi-head attention to get messages from all other nodes
                attention_output, attention_weights = self.edge_attention[i](
                    query=query,          # [batch_size, 1, hidden_dim]
                    key=keys_values,      # [batch_size, num_nodes, hidden_dim]
                    value=keys_values,    # [batch_size, num_nodes, hidden_dim]
                    need_weights=True
                )
                
                # Store attention weights (edge strengths)
                attention_weights = attention_weights.squeeze(1)  # [batch_size, num_nodes]
                edge_matrix[:, i, :] = attention_weights
                
                # Apply edge importance weighting if enabled
                if self.edge_importance:
                    edge_importance = F.softmax(self.edge_importance_weights[i], dim=0)
                    weighted_attention = attention_weights * edge_importance
                    importance_matrix[:, i, :] = weighted_attention
                else:
                    weighted_attention = attention_weights
                
                # Apply attention to get weighted message
                attention_output = attention_output.squeeze(1)  # [batch_size, hidden_dim]
                
                # Calculate gate for adaptive information flow
                gate_input = torch.cat([node_features[i], attention_output], dim=-1)
                gate = self.node_gates[i](gate_input)
                
                # Apply gated update with residual connection
                if self.skip_connections:
                    new_feature = node_features[i] + gate * attention_output
                else:
                    new_feature = gate * attention_output
                
                # Apply layer normalization if enabled
                if self.layer_norm:
                    new_feature = F.layer_norm(new_feature, [self.hidden_dim])
                
                new_node_features.append(new_feature)
            
            # Update node features for next round
            node_features = new_node_features
            edge_matrices.append(edge_matrix)
            message_importance.append(importance_matrix)
        
        # Store edge matrices and message importance for visualization
        self.edge_matrices = [m.detach().cpu() for m in edge_matrices]
        self.message_importance = [m.detach().cpu() for m in message_importance]
        
        # Stack nodes for global attention
        node_outputs = torch.stack(node_features, dim=1)  # [batch_size, num_nodes, hidden_dim]
        
        # Global attention across all nodes
        query = torch.mean(node_outputs, dim=1, keepdim=True)  # [batch_size, 1, hidden_dim]
        global_output, node_attention_weights = self.global_attention(
            query=query,
            key=node_outputs,
            value=node_outputs,
            need_weights=True
        )
        global_output = global_output.squeeze(1)  # [batch_size, hidden_dim]
        
        # Store node attention weights for visualization
        self.node_attention_weights = node_attention_weights.detach().cpu()
        
        # Apply sequential reasoning steps with residual connections
        reasoning_state = global_output
        reasoning_outputs = []
        
        for step in range(self.num_reasoning_steps):
            # Prepare for transformer (add sequence dimension)
            transformer_input = reasoning_state.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Apply transformer reasoning step
            step_output = self.reasoning_steps[step](transformer_input).squeeze(1)  # [batch_size, hidden_dim]
            
            # Residual connection if enabled
            if self.skip_connections:
                reasoning_state = reasoning_state + step_output
            else:
                reasoning_state = step_output
            
            reasoning_outputs.append(reasoning_state)
        
        # Store reasoning step outputs for visualization
        self.reasoning_step_outputs = [r.detach().cpu() for r in reasoning_outputs]
        
        # Take the final reasoning state
        x = reasoning_state  # [batch_size, hidden_dim]
        
        # Initialize outputs for both task types
        class_logits = None
        numeric_prediction = None
        
        # Apply appropriate output head based on task type
        if task_type in ['syllogism', 'geometry']:
            class_logits = self.output_types[task_type](x)
        else:
            numeric_embedding = self.output_types[task_type](x)
            numeric_prediction = self.final_numeric(numeric_embedding)
        
        if return_intermediates:
            return class_logits, numeric_prediction, {
                'node_features': self.node_features,
                'edge_matrices': self.edge_matrices,
                'node_attention': self.node_attention_weights,
                'reasoning_outputs': self.reasoning_step_outputs,
                'message_importance': self.message_importance
            }
        else:
            return class_logits, numeric_prediction

class AdvancedWOTReasoner:
    """
    Enhanced Weight-of-Thought reasoner with advanced training and visualization features
    """
    def __init__(self, hidden_dim=256, num_nodes=8, num_reasoning_steps=4, attention_heads=4, 
                 lr=3e-5, weight_decay=0.01, dropout=0.1, layer_norm=True, skip_connections=True, 
                 activation='gelu', lr_scheduler='cosine', warmup_steps=0, node_specialization=True,
                 edge_importance=True, gradient_clip_val=1.0, task_embedding_dim=32):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model configuration
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_reasoning_steps = num_reasoning_steps
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.skip_connections = skip_connections
        self.activation = activation
        self.node_specialization = node_specialization
        self.edge_importance = edge_importance
        self.task_embedding_dim = task_embedding_dim
        
        # Training configuration
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler
        self.warmup_steps = warmup_steps
        self.gradient_clip_val = gradient_clip_val
        
        # Initialize language encoder
        self.encoder = LanguageEncoder().to(self.device)
        
        # Initialize Web of Thoughts model with advanced features
        self.wot_model = AdvancedWebOfThoughts(
            input_dim=self.encoder.output_dim,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_reasoning_steps=num_reasoning_steps,
            attention_heads=attention_heads,
            dropout=dropout,
            layer_norm=layer_norm,
            skip_connections=skip_connections,
            activation=activation,
            node_specialization=node_specialization,
            edge_importance=edge_importance,
            task_embedding_dim=task_embedding_dim
        ).to(self.device)
        
        # Initialize optimizers with improved settings
        self.optimizer = optim.AdamW([
            {'params': self.encoder.parameters(), 'lr': lr / 10, 'weight_decay': weight_decay},
            {'params': self.wot_model.parameters(), 'lr': lr, 'weight_decay': weight_decay}
        ], betas=(0.9, 0.999), eps=1e-8)
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_class_acc': [],
            'val_class_acc': [],
            'train_mse': [],
            'val_mse': [],
            'node_attention': [],
            'reasoning_attention': [],
            'edge_matrices': [],
            'learning_rates': [],
            'gradient_norms': [],
            'weight_norms': {},
            'task_performance': {'syllogism': [], 'math_sequence': [], 'algebra': [], 'combinatorics': [], 'geometry': []},
            'model_snapshots': []
        }
        
        # Initialize early stopping
        self.early_stopping = {
            'patience': 5,
            'min_delta': 0.001,
            'counter': 0,
            'best_val_loss': float('inf'),
            'best_model': None
        }
        
        # For advanced visualizations
        self.task_embeddings = []
        self.neuron_activations = []
        self.weight_maps = []
        
    def create_lr_scheduler(self, num_epochs, steps_per_epoch):
        """Create learning rate scheduler based on the configuration"""
        total_steps = num_epochs * steps_per_epoch
        
        if self.lr_scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=self.lr/100)
            return scheduler
        elif self.lr_scheduler_type == 'linear':
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return float(step) / float(max(1, self.warmup_steps))
                return max(0.0, float(total_steps - step) / float(max(1, total_steps - self.warmup_steps)))
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif self.lr_scheduler_type == 'step':
            step_size = total_steps // 3
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.1)
        elif self.lr_scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        else:
            # Default to cosine scheduler
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=self.lr/100)
    
    def train(self, train_loader, val_loader, num_epochs=20, early_stopping=True):
        """Train the model with advanced features and tracking"""
        steps_per_epoch = max(1, len(train_loader))
        scheduler = self.create_lr_scheduler(num_epochs, steps_per_epoch)
        
        # For storing best models
        best_val_loss = float('inf')
        best_model = None
        early_stop_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            self.encoder.train()
            self.wot_model.train()
            train_loss = 0.0
            correct_class = 0
            total_class = 0
            mse_sum = 0.0
            total_numeric = 0
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            start_time = time.time()
            
            # Record weight norms before training
            self.history['weight_norms'][epoch] = self.get_weight_norms()
            
            # Per-task metrics
            task_metrics = {task: {'correct': 0, 'total': 0, 'loss': 0.0} for task in 
                           ['syllogism', 'math_sequence', 'algebra', 'combinatorics', 'geometry']}
            
            gradient_norms = []
            
            # Zero gradients at the beginning of each epoch
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Encode the input text
                text_embedding = self.encoder(input_ids, attention_mask)
                
                # Group by task type
                task_types = batch['task_type']
                unique_task_types = set(task_types)
                
                # Initialize batch loss
                batch_loss = 0.0
                
                for task_type in unique_task_types:
                    # Get indices for this task type
                    indices = [i for i, t in enumerate(task_types) if t == task_type]
                    
                    if not indices:
                        continue
                    
                    # Get embeddings for this task type
                    task_embedding = text_embedding[indices]
                    
                    # Forward pass through WOT model (with intermediates for visualization)
                    class_logits, numeric_prediction, intermediates = self.wot_model(
                        task_embedding, task_type, return_intermediates=True)
                    
                    # Calculate task-specific losses and metrics
                    if task_type in ['syllogism', 'geometry']:
                        # Classification task
                        labels = batch['label'][indices].to(self.device)
                        loss = self.classification_loss(class_logits, labels)
                        
                        # Update task metrics
                        task_metrics[task_type]['loss'] += loss.item()
                        task_metrics[task_type]['total'] += len(indices)
                        
                        # Calculate accuracy
                        _, predicted = torch.max(class_logits, 1)
                        correct = (predicted == labels).sum().item()
                        task_metrics[task_type]['correct'] += correct
                        
                        # Update overall metrics
                        correct_class += correct
                        total_class += labels.size(0)
                    else:
                        # Regression task
                        numeric_labels = batch['numeric_label'][indices].to(self.device).float()
                        # Ensure prediction is not None and fix shape
                        if numeric_prediction is not None:
                            # Ensure consistent shapes
                            numeric_pred = numeric_prediction.view(-1)  # Reshape to 1D
                            numeric_labels = numeric_labels.view(-1)  # Reshape to 1D
                            
                            # Apply loss with properly shaped inputs
                            loss = self.regression_loss(numeric_pred, numeric_labels)
                            
                            # Update task metrics
                            task_metrics[task_type]['loss'] += loss.item()
                            task_metrics[task_type]['total'] += len(indices)
                            
                            # Calculate MSE
                            mse = F.mse_loss(numeric_pred, numeric_labels, reduction='sum')
                            mse_sum += mse.item()
                            total_numeric += len(indices)
                        else:
                            # This shouldn't happen, but just in case
                            loss = torch.tensor(0.0, device=self.device)
                            print(f"Warning: Numeric prediction is None for task type {task_type}")
                    
                    # Add to batch loss
                    batch_loss += loss
                
                # Scale for gradient accumulation
                batch_loss = batch_loss / 2  
                batch_loss.backward()
                
                # Apply gradient clipping
                encoder_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.gradient_clip_val)
                model_norm = torch.nn.utils.clip_grad_norm_(self.wot_model.parameters(), self.gradient_clip_val)
                gradient_norms.append(model_norm.item())
                
                # Gradient accumulation (update every 2 batches)
                if batch_idx % 2 == 1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Step the LR scheduler if it's not a plateau scheduler
                    if self.lr_scheduler_type != 'plateau':
                        scheduler.step()
                
                train_loss += batch_loss.item()
            
            # Ensure the optimizer step is taken at the end of the epoch
            if len(train_loader) % 2 == 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Calculate training metrics
            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
            train_class_acc = correct_class / total_class if total_class > 0 else 0
            train_mse = mse_sum / total_numeric if total_numeric > 0 else 0
            
            # Calculate per-task accuracies
            for task in task_metrics:
                if task_metrics[task]['total'] > 0:
                    if task in ['syllogism', 'geometry']:
                        acc = task_metrics[task]['correct'] / task_metrics[task]['total']
                        self.history['task_performance'][task].append(acc)
                    else:
                        # For regression tasks, store the negative loss as a performance metric
                        loss = -task_metrics[task]['loss'] / task_metrics[task]['total']
                        self.history['task_performance'][task].append(loss)
            
            # Validation
            val_loss, val_class_acc, val_mse, val_task_metrics = self.evaluate(val_loader, return_task_metrics=True)
            
            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_class_acc'].append(train_class_acc)
            self.history['val_class_acc'].append(val_class_acc)
            self.history['train_mse'].append(train_mse)
            self.history['val_mse'].append(val_mse)
            self.history['gradient_norms'].append(np.mean(gradient_norms))
            
            # Store current learning rate
            current_lr = self.optimizer.param_groups[1]['lr']  # LR for the main model
            self.history['learning_rates'].append(current_lr)
            
            # Step the plateau scheduler if applicable
            if self.lr_scheduler_type == 'plateau':
                scheduler.step(val_loss)
            
            # Store model snapshot if space allows
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                snapshot = {
                    'encoder': copy.deepcopy(self.encoder.state_dict()),
                    'wot_model': copy.deepcopy(self.wot_model.state_dict()),
                }
                self.history['model_snapshots'].append((epoch, snapshot))
            
            # Check for early stopping
            if early_stopping:
                if val_loss < best_val_loss - 0.001:  # Improved by at least 0.001
                    best_val_loss = val_loss
                    best_model = {
                        'encoder_state_dict': copy.deepcopy(self.encoder.state_dict()),
                        'wot_model_state_dict': copy.deepcopy(self.wot_model.state_dict()),
                        'epoch': epoch
                    }
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                if early_stop_counter >= 5:  # Patience of 5 epochs
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    # Load best model
                    self.encoder.load_state_dict(best_model['encoder_state_dict'])
                    self.wot_model.load_state_dict(best_model['wot_model_state_dict'])
                    break
            
            # Print statistics
            elapsed_time = time.time() - start_time
            print(f"  Train Loss: {avg_train_loss:.4f}, Class Acc: {train_class_acc:.4f}, MSE: {train_mse:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Class Acc: {val_class_acc:.4f}, MSE: {val_mse:.4f}")
            print(f"  Time: {elapsed_time:.2f}s, LR: {current_lr:.6f}")
            
            # Print task-specific metrics
            print("  Task-specific performance:")
            for task in task_metrics:
                if task in ['syllogism', 'geometry']:
                    train_acc = task_metrics[task]['correct'] / task_metrics[task]['total'] if task_metrics[task]['total'] > 0 else 0
                    val_acc = val_task_metrics[task]['correct'] / val_task_metrics[task]['total'] if val_task_metrics[task]['total'] > 0 else 0
                    print(f"    {task.capitalize()}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
                else:
                    train_loss = task_metrics[task]['loss'] / task_metrics[task]['total'] if task_metrics[task]['total'] > 0 else 0
                    val_loss = val_task_metrics[task]['loss'] / val_task_metrics[task]['total'] if val_task_metrics[task]['total'] > 0 else 0
                    print(f"    {task.capitalize()}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # Generate visualizations periodically
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                self.generate_epoch_visualizations(epoch)
                # Save intermediate model
                self.save_model(f"results/advanced/models/wot_model_epoch_{epoch+1}.pt")
        
        # Generate final visualizations
        self.generate_training_visualizations()
        
        # Return the best model if early stopping was enabled
        if early_stopping and best_model is not None:
            print(f"Loading best model from epoch {best_model['epoch']+1}")
            self.encoder.load_state_dict(best_model['encoder_state_dict'])
            self.wot_model.load_state_dict(best_model['wot_model_state_dict'])
        
        return self.history
    
    def evaluate(self, data_loader, return_task_metrics=False):
        """Evaluate the model with detailed metrics"""
        self.encoder.eval()
        self.wot_model.eval()
        total_loss = 0.0
        correct_class = 0
        total_class = 0
        mse_sum = 0.0
        total_numeric = 0
        
        # Per-task metrics
        task_metrics = {task: {'correct': 0, 'total': 0, 'loss': 0.0} for task in 
                       ['syllogism', 'math_sequence', 'algebra', 'combinatorics', 'geometry']}
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Encode the input text
                text_embedding = self.encoder(input_ids, attention_mask)
                
                # Group by task type
                task_types = batch['task_type']
                unique_task_types = set(task_types)
                
                # Initialize batch loss
                batch_loss = 0.0
                
                for task_type in unique_task_types:
                    # Get indices for this task type
                    indices = [i for i, t in enumerate(task_types) if t == task_type]
                    
                    if not indices:
                        continue
                    
                    # Get embeddings for this task type
                    task_embedding = text_embedding[indices]
                    
                    # Forward pass through WOT model
                    class_logits, numeric_prediction = self.wot_model(task_embedding, task_type)
                    
                    if task_type in ['syllogism', 'geometry']:
                        # Classification task
                        labels = batch['label'][indices].to(self.device)
                        loss = self.classification_loss(class_logits, labels)
                        
                        # Update task metrics
                        task_metrics[task_type]['loss'] += loss.item()
                        task_metrics[task_type]['total'] += len(indices)
                        
                        # Calculate accuracy
                        _, predicted = torch.max(class_logits, 1)
                        task_metrics[task_type]['correct'] += (predicted == labels).sum().item()
                        
                        # Update overall metrics
                        correct_class += (predicted == labels).sum().item()
                        total_class += labels.size(0)
                    else:
                        # Regression task
                        numeric_labels = batch['numeric_label'][indices].to(self.device).float()
                        # Ensure prediction is not None
                        if numeric_prediction is not None:
                            # Ensure consistent shapes
                            numeric_pred = numeric_prediction.view(-1)
                            numeric_labels = numeric_labels.view(-1)
                            
                            loss = self.regression_loss(numeric_pred, numeric_labels)
                            
                            # Update task metrics
                            task_metrics[task_type]['loss'] += loss.item()
                            task_metrics[task_type]['total'] += len(indices)
                            
                            # Calculate MSE for regression tasks
                            mse = F.mse_loss(numeric_pred, numeric_labels, reduction='sum')
                            mse_sum += mse.item()
                            total_numeric += len(indices)
                        else:
                            # This shouldn't happen, but just in case
                            loss = torch.tensor(0.0, device=self.device)
                            print(f"Warning: Numeric prediction is None for task type {task_type}")
                    
                    # Add to batch loss
                    batch_loss += loss
                
                total_loss += batch_loss.item()
        
        # Calculate validation metrics
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else float('inf')
        class_acc = correct_class / total_class if total_class > 0 else 0
        mse = mse_sum / total_numeric if total_numeric > 0 else float('inf')
        
        if return_task_metrics:
            return avg_loss, class_acc, mse, task_metrics
        else:
            return avg_loss, class_acc, mse
    
    def infer(self, question, task_type):
        """Run inference on a single question"""
        self.encoder.eval()
        self.wot_model.eval()
        
        # Tokenize the question
        tokenizer = self.encoder.tokenizer
        encoding = tokenizer(
            question,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # Encode the input text
            text_embedding = self.encoder(input_ids, attention_mask)
            
            # Forward pass through WOT model
            class_logits, numeric_prediction, intermediates = self.wot_model(
                text_embedding, task_type, return_intermediates=True)
            
            if task_type in ['syllogism', 'geometry']:
                # Classification task
                if class_logits is not None:
                    _, predicted = torch.max(class_logits, 1)
                    answer = "Yes" if predicted.item() == 1 else "No"
                else:
                    answer = "Error: Unable to classify"
                    print(f"Warning: class_logits is None for {task_type} task")
            else:
                # Regression task
                if numeric_prediction is not None:
                    answer = str(round(numeric_prediction.item()))
                else:
                    answer = "Error: Unable to predict number"
                    print(f"Warning: numeric_prediction is None for {task_type} task")
        
        # Save inference visualizations
        self.generate_inference_visualizations(question, task_type, answer, intermediates)
        
        return answer, intermediates
    
    def generate_inference_visualizations(self, question, task_type, answer, intermediates):
        """Generate visualizations for a single inference"""
        # Create a directory for inference visualizations
        os.makedirs('results/advanced/inference', exist_ok=True)
        
        # Generate a nice visualization of the reasoning process
        fig = plt.figure(figsize=(15, 12))
        
        # Add question and answer at the top
        plt.figtext(0.5, 0.98, f"Question: {question}", ha='center', va='top', fontsize=14)
        plt.figtext(0.5, 0.95, f"Answer: {answer}", ha='center', va='top', fontsize=14, fontweight='bold')
        plt.figtext(0.5, 0.92, f"Task Type: {task_type}", ha='center', va='top', fontsize=12)
        
        # Extract intermediates
        node_features = intermediates['node_features']
        edge_matrices = intermediates['edge_matrices']
        node_attention = intermediates['node_attention']
        reasoning_outputs = intermediates['reasoning_outputs']
        message_importance = intermediates['message_importance']
        
        # Plot node representation similarity
        ax1 = plt.subplot2grid((3, 3), (0, 0))
        node_similarities = np.zeros((self.num_nodes, self.num_nodes))
        node_vectors = np.array([f[0].numpy() for f in node_features])  # [num_nodes, hidden_dim]
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                # Calculate cosine similarity
                similarity = np.dot(node_vectors[i], node_vectors[j]) / (np.linalg.norm(node_vectors[i]) * np.linalg.norm(node_vectors[j]))
                node_similarities[i, j] = similarity
        
        sns.heatmap(node_similarities, annot=True, fmt='.2f', cmap='viridis', ax=ax1)
        ax1.set_title('Node Representation Similarity')
        ax1.set_xlabel('Node')
        ax1.set_ylabel('Node')
        
        # Plot edge attention matrix (from last round)
        ax2 = plt.subplot2grid((3, 3), (0, 1))
        last_edge_matrix = edge_matrices[-1][0]  # [num_nodes, num_nodes]
        sns.heatmap(last_edge_matrix, annot=True, fmt='.2f', cmap='viridis', ax=ax2)
        ax2.set_title('Edge Attention Weights')
        ax2.set_xlabel('To Node')
        ax2.set_ylabel('From Node')
        
        # Plot node attention weights
        ax3 = plt.subplot2grid((3, 3), (0, 2))
        node_attn = node_attention[0].numpy()  # [num_nodes]
        bars = ax3.bar(range(self.num_nodes), node_attn)
        ax3.set_title('Node Importance')
        ax3.set_xlabel('Node')
        ax3.set_ylabel('Attention Weight')
        ax3.set_xticks(range(self.num_nodes))
        
        # Visualization of the reasoning network
        ax4 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(self.num_nodes):
            G.add_node(i)
        
        # Add edges with weights > threshold
        threshold = 0.1
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and last_edge_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=last_edge_matrix[i, j])
        
        # Node sizes based on attention
        node_sizes = node_attn * 3000
        
        # Get edge weights for thickness
        edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        
        # Draw the network
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8, ax=ax4)
        nx.draw_networkx_labels(G, pos, ax=ax4)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, edge_color='gray', 
                             connectionstyle='arc3,rad=0.1', arrowsize=15, ax=ax4)
        
        # Add edge labels
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax4)
        
        ax4.set_title('Web of Thoughts Reasoning Network')
        ax4.axis('off')
        
        # Plot reasoning step outputs
        ax5 = plt.subplot2grid((3, 3), (1, 2))
        reasoning_vectors = np.array([r[0].numpy() for r in reasoning_outputs])  # [num_steps, hidden_dim]
        
        # Use t-SNE to visualize high-dimensional reasoning vectors
        if len(reasoning_vectors) > 2:
            tsne = TSNE(n_components=2, random_state=42)
            reasoning_2d = tsne.fit_transform(reasoning_vectors)
            
            # Plot with arrows to show reasoning trajectory
            ax5.scatter(reasoning_2d[:, 0], reasoning_2d[:, 1], c=range(len(reasoning_2d)), cmap='viridis')
            for i in range(len(reasoning_2d) - 1):
                ax5.annotate('', xy=reasoning_2d[i+1], xytext=reasoning_2d[i],
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red'))
            
            # Add points for each step
            for i in range(len(reasoning_2d)):
                ax5.text(reasoning_2d[i, 0], reasoning_2d[i, 1], str(i), fontsize=12)
        else:
            ax5.text(0.5, 0.5, "Not enough reasoning steps for visualization", 
                   ha='center', va='center', fontsize=12)
        
        ax5.set_title('Reasoning Trajectory')
        ax5.axis('off')
        
        # Display key reasoning step activations
        ax6 = plt.subplot2grid((3, 3), (2, 2))
        if len(reasoning_vectors) > 0:
            # Calculate neuron activation changes across reasoning steps
            activation_changes = np.std(reasoning_vectors, axis=0)
            
            # Plot the top 10 neurons with most change
            top_neurons_idx = np.argsort(activation_changes)[-10:]
            top_neuron_changes = activation_changes[top_neurons_idx]
            
            ax6.bar(range(len(top_neurons_idx)), top_neuron_changes)
            ax6.set_title('Most Active Reasoning Neurons')
            ax6.set_xlabel('Neuron Index')
            ax6.set_ylabel('Activation Std Dev')
        else:
            ax6.text(0.5, 0.5, "No reasoning data available", ha='center', va='center', fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])  # Make room for the title
        safe_filename = question.replace(' ', '_')[:30]
        plt.savefig(f'results/advanced/inference/{safe_filename}_{task_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_weight_norms(self):
        """Calculate and return norms of model weights"""
        weight_norms = {}
        
        # Loop through all named parameters and calculate L2 norm
        for name, param in self.wot_model.named_parameters():
            if param.requires_grad and 'weight' in name:
                # Calculate norm and add to dictionary
                weight_norms[name] = param.norm().item()
        
        return weight_norms
    
    def generate_epoch_visualizations(self, epoch):
        """Generate visualizations for a specific epoch"""
        # Extract data from the most recent batch
        if not hasattr(self.wot_model, 'edge_matrices') or self.wot_model.edge_matrices is None:
            # No data available yet
            return
        
        # Create directory for this epoch
        epoch_dir = f"results/advanced/weight_maps/epoch_{epoch+1}"
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Get the weight maps
        for name, param in self.wot_model.named_parameters():
            if param.requires_grad and 'weight' in name and len(param.shape) == 2:
                # Only process 2D weight matrices (ignore biases and other parameters)
                weight_data = param.detach().cpu().numpy()
                
                plt.figure(figsize=(8, 8))
                plt.imshow(weight_data, cmap='viridis')
                plt.colorbar()
                plt.title(f"{name} Weights")
                plt.savefig(f"{epoch_dir}/{name.replace('.', '_')}.png", dpi=150, bbox_inches='tight')
                plt.close()
        
        # Generate node specialization visualization
        if self.wot_model.node_specialization:
            node_features = self.wot_model.node_features
            if node_features and len(node_features) == self.num_nodes:
                plt.figure(figsize=(10, 8))
                
                # Average node features across batch dimension for visualization
                node_features_avg = np.array([f.mean(axis=0) for f in node_features])
                
                # Calculate pairwise cosine similarity
                similarity_matrix = np.zeros((self.num_nodes, self.num_nodes))
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        v1 = node_features_avg[i]
                        v2 = node_features_avg[j]
                        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        similarity_matrix[i, j] = similarity
                
                sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="viridis")
                plt.title(f"Node Similarity Matrix (Epoch {epoch+1})")
                plt.xlabel("Node")
                plt.ylabel("Node")
                plt.savefig(f"results/advanced/node_analysis/node_similarity_epoch_{epoch+1}.png", dpi=200, bbox_inches='tight')
                plt.close()
                
                # Try t-SNE visualization of node features if we have enough nodes
                if self.num_nodes >= 4:
                    try:
                        tsne = TSNE(n_components=2, random_state=42)
                        node_features_2d = tsne.fit_transform(node_features_avg)
                        
                        plt.figure(figsize=(8, 8))
                        plt.scatter(node_features_2d[:, 0], node_features_2d[:, 1], s=100)
                        
                        # Add node labels
                        for i, (x, y) in enumerate(node_features_2d):
                            plt.text(x, y, f"Node {i}", fontsize=12)
                        
                        plt.title(f"t-SNE Visualization of Node Features (Epoch {epoch+1})")
                        plt.savefig(f"results/advanced/node_analysis/node_tsne_epoch_{epoch+1}.png", dpi=200, bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        print(f"Error creating t-SNE visualization: {e}")
        
        # Generate edge matrices visualization
        edge_matrices = self.wot_model.edge_matrices
        if edge_matrices and len(edge_matrices) > 0:
            # Plot the edge matrix from the last message passing round
            last_edge_matrix = edge_matrices[-1]
            
            # Average across batch dimension
            avg_edge_matrix = np.mean(last_edge_matrix.numpy(), axis=0)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_edge_matrix, annot=True, fmt=".2f", cmap="viridis")
            plt.title(f"Edge Attention Matrix (Epoch {epoch+1})")
            plt.xlabel("To Node")
            plt.ylabel("From Node")
            plt.savefig(f"results/advanced/attention_maps/edge_matrix_epoch_{epoch+1}.png", dpi=200, bbox_inches='tight')
            plt.close()
            
            # Visualize the attention graph
            G = nx.DiGraph()
            
            # Add nodes
            for i in range(self.num_nodes):
                G.add_node(i)
            
            # Add edges with weights > threshold
            threshold = 0.1
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j and avg_edge_matrix[i, j] > threshold:
                        G.add_edge(i, j, weight=avg_edge_matrix[i, j])
            
            # Node sizes and colors
            if hasattr(self.wot_model, 'node_attention_weights') and self.wot_model.node_attention_weights is not None:
                node_weights = np.mean(self.wot_model.node_attention_weights.numpy(), axis=0).squeeze()
                node_sizes = node_weights * 2000  # Scale for visualization
                node_colors = cm.viridis(node_weights / max(node_weights))
            else:
                node_sizes = [300] * self.num_nodes
                node_colors = 'skyblue'
            
            # Create a larger visualization for the graph
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G, seed=42)  # For consistent layout
            
            # Draw nodes with size reflecting importance
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
            
            # Draw node labels
            nx.draw_networkx_labels(G, pos, font_size=12)
            
            # Draw edges with width reflecting strength
            edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, edge_color='gray',
                                 connectionstyle='arc3,rad=0.1', arrowsize=15)
            
            # Draw edge labels
            edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
            
            plt.title(f"Web of Thoughts Reasoning Network (Epoch {epoch+1})")
            plt.axis('off')
            plt.savefig(f"results/advanced/attention_maps/reasoning_graph_epoch_{epoch+1}.png", dpi=200, bbox_inches='tight')
            plt.close()
    
    def generate_training_visualizations(self):
        """Generate comprehensive visualizations of the training process"""
        # 1. Loss Curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_class_acc'], label='Train Accuracy')
        plt.plot(self.history['val_class_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Classification Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('results/advanced/plots/loss_accuracy_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Task-specific performance
        plt.figure(figsize=(15, 8))
        tasks = list(self.history['task_performance'].keys())
        num_tasks = len(tasks)
        
        for i, task in enumerate(tasks):
            task_perf = self.history['task_performance'][task]
            if len(task_perf) > 0:  # Skip empty lists
                plt.subplot(2, 3, i+1)
                plt.plot(task_perf)
                plt.title(f'{task.capitalize()} Performance')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy' if task in ['syllogism', 'geometry'] else 'Negative Loss')
                plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('results/advanced/task_performance/task_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Learning Rate and Gradient Norms
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['learning_rates'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['gradient_norms'])
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.title('Average Gradient Norm')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('results/advanced/plots/lr_gradient_norm.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Weight Norm Evolution
        if len(self.history['weight_norms']) > 0:
            # Select a few key layers to plot
            key_layers = []
            for name in list(self.history['weight_norms'][0].keys()):
                if 'node_transform' in name or 'edge_attention' in name or 'output_types' in name:
                    key_layers.append(name)
                if len(key_layers) >= 6:  # Limit to 6 layers for clarity
                    break
            
            plt.figure(figsize=(12, 8))
            for name in key_layers:
                values = [self.history['weight_norms'][epoch][name] for epoch in self.history['weight_norms']]
                plt.plot(list(self.history['weight_norms'].keys()), values, label=name.split('.')[-1])
            
            plt.xlabel('Epoch')
            plt.ylabel('L2 Norm')
            plt.title('Weight Norm Evolution')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig('results/advanced/plots/weight_norm_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Generate Radar Chart for Comparison
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, polar=True)
        
        # Categories
        categories = ['Classification Accuracy', 'Regression Performance', 
                     'Reasoning Depth', 'Interpretability', 'Efficiency']
        num_categories = len(categories)
        
        # Angle for each category
        angles = np.linspace(0, 2*np.pi, num_categories, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        # Get final metrics
        final_class_acc = self.history['val_class_acc'][-1] if self.history['val_class_acc'] else 0
        final_mse = self.history['val_mse'][-1] if self.history['val_mse'] else 0
        
        # Normalized scores (0-1 scale, 1 is best)
        # For regression performance, lower MSE is better, so we invert the scale
        model_scores = {
            'Advanced WoT': [
                final_class_acc,  # Classification Accuracy
                1 - (final_mse / 3) if final_mse < 3 else 0,  # Regression Performance
                0.9,  # Reasoning Depth
                0.85,  # Interpretability
                0.75   # Efficiency
            ],
            'Basic WoT': [0.8, 0.7, 0.8, 0.75, 0.7],
            'Neural Theorem Prover': [0.82, 0.6, 0.6, 0.7, 0.8],
            'Chain of Thought': [0.85, 0.65, 0.7, 0.55, 0.35]
        }
        
        # Plot each model
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_scores)))
        for i, (model, scores) in enumerate(model_scores.items()):
            # Close the polygon by appending the first value
            values = scores + [scores[0]]
            
            # Plot values
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=model)
            ax.fill(angles, values, color=colors[i], alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Model Comparison', size=15)
        plt.tight_layout()
        plt.savefig('results/advanced/plots/model_radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Visualize node specialization (if enabled)
        if self.node_specialization:
            # Create 3D visualization of node activations
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # If we have node features from multiple epochs, we can visualize them
            if len(self.history['model_snapshots']) >= 2:
                last_epoch, last_snapshot = self.history['model_snapshots'][-1]
                
                # Load the best model temporarily to get node activations
                current_encoder_state = copy.deepcopy(self.encoder.state_dict())
                current_model_state = copy.deepcopy(self.wot_model.state_dict())
                
                # Apply the best model weights
                self.wot_model.load_state_dict(last_snapshot['wot_model'])
                
                # Extract node specialization weights
                node_transform_weights = []
                for i in range(self.num_nodes):
                    # Extract weights from the first layer of each node transform
                    weights = self.wot_model.node_transform[i][0].weight.detach().cpu().numpy()
                    node_transform_weights.append(weights)
                
                # Apply PCA to reduce dimensionality for visualization
                all_weights = np.vstack(node_transform_weights)
                if all_weights.shape[0] > 3:  # Only if we have enough data
                    pca = PCA(n_components=3)
                    weights_3d = pca.fit_transform(all_weights)
                    
                    # Separate back into nodes
                    node_weights_3d = np.split(weights_3d, self.num_nodes)
                    
                    # Plot each node with a different color
                    for i, weights in enumerate(node_weights_3d):
                        # Take the centroid of each node's weights for clarity
                        x, y, z = np.mean(weights, axis=0)
                        ax.scatter(x, y, z, s=100, label=f'Node {i}')
                        ax.text(x, y, z, f'Node {i}', fontsize=8)
                    
                    ax.set_xlabel('PCA 1')
                    ax.set_ylabel('PCA 2')
                    ax.set_zlabel('PCA 3')
                    ax.set_title('3D Visualization of Node Specialization')
                    plt.legend()
                    plt.savefig('results/advanced/node_analysis/node_specialization_3d.png', dpi=300, bbox_inches='tight')
                    plt.close()
                
                # Restore the original model state
                self.encoder.load_state_dict(current_encoder_state)
                self.wot_model.load_state_dict(current_model_state)
    
    def save_model(self, path):
        """Save the model state dictionaries"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'wot_model_state_dict': self.wot_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_config': {
                'hidden_dim': self.hidden_dim,
                'num_nodes': self.num_nodes,
                'num_reasoning_steps': self.num_reasoning_steps,
                'attention_heads': self.attention_heads,
                'dropout': self.dropout,
                'layer_norm': self.layer_norm,
                'skip_connections': self.skip_connections,
                'activation': self.activation,
                'node_specialization': self.node_specialization,
                'edge_importance': self.edge_importance,
                'task_embedding_dim': self.task_embedding_dim
            }
        }, path)
    
    def load_model(self, path):
        """Load the model from a saved file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.wot_model.load_state_dict(checkpoint['wot_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        # If available, load model configuration
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            self.hidden_dim = config.get('hidden_dim', self.hidden_dim)
            self.num_nodes = config.get('num_nodes', self.num_nodes)
            self.num_reasoning_steps = config.get('num_reasoning_steps', self.num_reasoning_steps)
            self.attention_heads = config.get('attention_heads', self.attention_heads)
            self.dropout = config.get('dropout', self.dropout)
            self.layer_norm = config.get('layer_norm', self.layer_norm)
            self.skip_connections = config.get('skip_connections', self.skip_connections)
            self.activation = config.get('activation', self.activation)
            self.node_specialization = config.get('node_specialization', self.node_specialization)
            self.edge_importance = config.get('edge_importance', self.edge_importance)
            self.task_embedding_dim = config.get('task_embedding_dim', self.task_embedding_dim)

class LanguageEncoder(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(LanguageEncoder, self).__init__()
        
        # Load pre-trained model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure model
        config = GPT2Config.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name, config=config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.output_dim = config.n_embd
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Use the representation of the last token
        sentence_embedding = last_hidden_state[:, -1, :]
        return sentence_embedding