import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random
import pygame
import math
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Chain-of-Thought Reasoning Network with Transformer Architecture
class ReasoningNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_heads=4, num_layers=3):
        super(ReasoningNetwork, self).__init__()
        
        # Dimensions
        self.d_model = 256
        self.state_size = state_size
        self.action_size = action_size
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(state_size, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU()
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reasoning modules
        self.reasoning_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])
        
        # Output heads
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        # Auxiliary prediction heads for better representation learning
        self.next_state_predictor = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, state_size)
        )
        
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Store attention weights
        self.last_attention_weights = None
        
    def forward(self, state):
        # Initial embedding
        x = self.input_embedding(state)
        
        # Add positional encoding
        batch_size = x.shape[0]
        pos = torch.arange(0, self.d_model).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        x = x + pos
        
        # Transformer encoding
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        
        # Multi-step reasoning
        reasoning_outputs = []
        for layer in self.reasoning_layers:
            x = layer(x)
            reasoning_outputs.append(x)
        
        # Combine reasoning steps with attention
        attention_weights = torch.softmax(torch.stack(reasoning_outputs).mean(dim=2), dim=0)
        x = (torch.stack(reasoning_outputs) * attention_weights.unsqueeze(-1)).sum(0)
        
        # Store attention weights for visualization
        self.last_attention_weights = attention_weights.detach().cpu().numpy()
        
        # Output heads
        value = self.value_head(x)
        policy = self.policy_head(x)
        next_state_pred = self.next_state_predictor(x)
        reward_pred = self.reward_predictor(x)
        
        return policy, value, next_state_pred, reward_pred, attention_weights

    def get_attention_weights(self):
        return self.last_attention_weights if self.last_attention_weights is not None else np.zeros((3, 3))

# Advanced Prioritized Experience Replay with Hindsight
class EnhancedReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.epsilon = 1e-6
        
    def push(self, state, action, reward, next_state, done, info=None):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
        
    def sample(self, batch_size):
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.epsilon

    def __len__(self):
        return len(self.buffer)

# Visualization Enhancement
class ResearchVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig = plt.figure(figsize=(20, 12))
        self.gs = GridSpec(3, 3, figure=self.fig)

        # Initialize subplots
        self.ax_reward = self.fig.add_subplot(self.gs[0, :2])
        self.ax_loss = self.fig.add_subplot(self.gs[1, :2])
        self.ax_attention = self.fig.add_subplot(self.gs[2, :2])
        self.ax_metrics = self.fig.add_subplot(self.gs[:, 2])
        
        # Data storage
        self.rewards_history = []
        self.losses_history = []
        self.attention_history = []
        self.metrics = {
            'success_rate': [],
            'avg_steps': [],
            'exploration_ratio': []
        }
        
        # Style configuration
        self.colors = sns.color_palette("husl", 8)
        self.setup_style()
        
    def setup_style(self):
        for ax in [self.ax_reward, self.ax_loss, self.ax_attention, self.ax_metrics]:
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
    def update(self, episode_reward, loss, attention_weights, metrics):
        # Update histories
        self.rewards_history.append(episode_reward)
        self.losses_history.append(loss)
        self.attention_history.append(attention_weights)
        for key in metrics:
            self.metrics[key].append(metrics[key])
            
        self.plot_all()
        
    def plot_all(self):
        # Clear all axes
        for ax in [self.ax_reward, self.ax_loss, self.ax_attention, self.ax_metrics]:
            ax.clear()
            
        # Plot reward
        sns.lineplot(data=self.rewards_history, ax=self.ax_reward, color=self.colors[0])
        self.ax_reward.set_title('Episode Rewards', fontsize=12, pad=20)
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        
        # Plot loss
        sns.lineplot(data=self.losses_history, ax=self.ax_loss, color=self.colors[1])
        self.ax_loss.set_title('Training Loss', fontsize=12, pad=20)
        self.ax_loss.set_xlabel('Training Step')
        self.ax_loss.set_ylabel('Loss')
        
        # Plot attention heatmap
        if self.attention_history:
            sns.heatmap(self.attention_history[-1], ax=self.ax_attention, 
                       cmap='viridis', cbar_kws={'label': 'Attention Weight'})
            self.ax_attention.set_title('Reasoning Step Attention', fontsize=12, pad=20)
            
        # Plot metrics
        for i, (key, values) in enumerate(self.metrics.items()):
            sns.lineplot(data=values, ax=self.ax_metrics, label=key, color=self.colors[i+2])
        self.ax_metrics.set_title('Training Metrics', fontsize=12, pad=20)
        self.ax_metrics.legend()
        
        plt.tight_layout()
        plt.pause(0.01)

# Navigation Environment
class NavigationEnv:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Navigation Environment")
        
        # Agent properties
        self.agent_pos = np.array([width/2, height/2])
        self.agent_vel = np.array([0.0, 0.0])
        self.agent_radius = 15
        
        # Goal properties
        self.goal_pos = self._random_position()
        self.goal_radius = 20
        
        # Obstacles
        self.num_obstacles = 5
        self.obstacles = [self._random_position() for _ in range(self.num_obstacles)]
        self.obstacle_radius = 30
        
        # State space: agent position (2), agent velocity (2), goal position (2)
        self.state_size = 6
        # Action space: acceleration in x and y directions
        self.action_size = 2
        
    def _random_position(self):
        return np.array([
            random.randint(50, self.width-50),
            random.randint(50, self.height-50)
        ])
        
    def reset(self):
        self.agent_pos = np.array([self.width/2, self.height/2])
        self.agent_vel = np.array([0.0, 0.0])
        self.goal_pos = self._random_position()
        self.obstacles = [self._random_position() for _ in range(self.num_obstacles)]
        return self._get_state()
        
    def _get_state(self):
        return np.concatenate([
            self.agent_pos,
            self.agent_vel,
            self.goal_pos
        ])
        
    def step(self, action):
        # Update agent position and velocity
        self.agent_vel += action
        self.agent_vel = np.clip(self.agent_vel, -5, 5)
        self.agent_pos += self.agent_vel
        
        # Boundary checking
        self.agent_pos = np.clip(self.agent_pos, 0, [self.width, self.height])
        
        # Calculate reward
        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -0.1  # Small negative reward for each step
        done = False
        
        # Check if reached goal
        if distance_to_goal < self.goal_radius:
            reward = 100
            done = True
            
        # Check collision with obstacles
        for obs_pos in self.obstacles:
            if np.linalg.norm(self.agent_pos - obs_pos) < self.obstacle_radius:
                reward = -50
                done = True
                
        return self._get_state(), reward, done, {}
        
    def render(self):
        self.screen.fill((255, 255, 255))

        # Draw obstacles
        for obs_pos in self.obstacles:
            pygame.draw.circle(self.screen, (255, 0, 0), obs_pos.astype(int), self.obstacle_radius)

        # Draw goal
        pygame.draw.circle(self.screen, (0, 255, 0), self.goal_pos.astype(int), self.goal_radius)

        # Draw agent
        pygame.draw.circle(self.screen, (0, 0, 255), self.agent_pos.astype(int), self.agent_radius)
        
        pygame.display.flip()
        
    def close(self):
        pygame.quit()

def train():
    # Initialize environment and agent
    env = NavigationEnv()
    state_size = env.state_size
    action_size = env.action_size
    agent = ReasoningNetwork(state_size, action_size)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    replay_buffer = EnhancedReplayBuffer()
    visualizer = ResearchVisualizer()
    
    num_episodes = 1000
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0  # For exploration
    epsilon_min = 0.01
    epsilon_decay = 0.995
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        episode_loss = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy, value, _, _, attention = agent(state_tensor)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = np.random.randn(2)  # Random action
            else:
                action_probs = torch.softmax(policy, dim=1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().detach().numpy()
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Train if enough samples
            if len(replay_buffer) > batch_size:
                samples, indices, weights = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*samples)

                # Convert to tensors
                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)
                weights = torch.FloatTensor(weights)

                # Get current Q values
                current_policy, current_value, _, _, _ = agent(states)
                current_q = current_value.squeeze()

                # Get next Q values
                with torch.no_grad():
                    _, next_value, _, _, _ = agent(next_states)
                    next_q = next_value.squeeze()

                # Calculate target Q values
                target_q = rewards + gamma * next_q * (1 - dones)

                # Calculate loss
                value_loss = (weights * (current_q - target_q) ** 2).mean()
                policy_loss = -action_dist.log_prob(action) * weights
                total_loss = value_loss + policy_loss.mean()
                
                # Update network
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Update priorities
                td_errors = (target_q - current_q).detach().numpy()
                replay_buffer.update_priorities(indices, td_errors)

                episode_loss += total_loss.item()
            
            state = next_state
            steps += 1
            env.render()
            
            
            # Update visualization
            metrics = {
                'success_rate': total_reward > 0,
                'avg_steps': steps,
                'exploration_ratio': epsilon
            }
            visualizer.update(total_reward, episode_loss, agent.get_attention_weights(), metrics)

            if done:
                print(f"Episode {episode}: Reward = {total_reward}, Steps = {steps}")
                break

    # Save model periodically
    if episode % 100 == 0:
        torch.save(agent.state_dict(), f'model_checkpoint_{episode}.pth')

pygame.quit()
