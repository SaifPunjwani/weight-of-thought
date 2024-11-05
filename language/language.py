import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import random
import seaborn as sns

# Set style for prettier plots
plt.style.use('seaborn-v0_8')  # Updated style name for newer matplotlib versions
sns.set_palette("husl")

class LanguageReasoner(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(LanguageReasoner, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Binary classification (valid/invalid reasoning)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = self.dropout(torch.relu(self.fc1(lstm_out[:, -1, :])))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc3(x)

class DynamicPlotter:
    def __init__(self):
        # Create figure with subplots
        self.fig = plt.figure(figsize=(15, 8))
        self.ax1 = self.fig.add_subplot(231)
        self.ax2 = self.fig.add_subplot(232)
        self.ax3 = self.fig.add_subplot(233)
        self.ax4 = self.fig.add_subplot(212)

        # Initialize data storage
        self.accuracy_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.confidence_scores = deque(maxlen=50)
        self.attention_weights = np.random.rand(10, 10)  # Example attention matrix
        
        # Style settings
        self.fig.patch.set_facecolor('#f0f0f0')
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#ffffff')
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.ion()  # Enable interactive mode

    def update(self, accuracy, loss, confidence, attention=None):
        # Update data
        self.accuracy_history.append(accuracy)
        self.loss_history.append(loss)
        self.confidence_scores.append(confidence)
        if attention is not None:
            self.attention_weights = attention

        # Clear previous plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()

        # Plot accuracy
        self.ax1.plot(list(self.accuracy_history), color='#2ecc71', linewidth=2)
        self.ax1.set_title('Accuracy Over Time', pad=10)
        self.ax1.set_ylim(0, 1)

        # Plot loss
        self.ax2.plot(list(self.loss_history), color='#e74c3c', linewidth=2)
        self.ax2.set_title('Loss Over Time', pad=10)

        # Plot confidence distribution
        sns.kdeplot(data=list(self.confidence_scores), ax=self.ax3, color='#3498db', fill=True)
        self.ax3.set_title('Confidence Distribution', pad=10)

        # Plot attention heatmap
        sns.heatmap(self.attention_weights, ax=self.ax4, cmap='viridis', 
                   cbar_kws={'label': 'Attention Weight'})
        self.ax4.set_title('Attention Visualization', pad=10)

        # Update layout and display
        plt.tight_layout()
        plt.pause(0.01)

# Example usage:
plotter = DynamicPlotter()

# Simulate some training data
for i in range(200):
    accuracy = 0.5 + 0.4 * (1 - np.exp(-i/50)) + random.uniform(-0.05, 0.05)
    loss = 1.0 * np.exp(-i/50) + random.uniform(-0.05, 0.05)
    confidence = random.betavariate(5, 2)
    attention = np.random.rand(10, 10)  # Random attention matrix for visualization
    
    plotter.update(accuracy, loss, confidence, attention)
    
plt.ioff()
plt.show()
