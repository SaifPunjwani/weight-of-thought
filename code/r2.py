import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random
import pygame
import math

# Deep Q-Network for Logical Reasoning
class DQNReasoner(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNReasoner, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Simple 2D navigation environment
class NavigationEnv:
    def __init__(self):
        self.width = 800
        self.height = 600
        self.agent_pos = [400, 300]
        self.target_pos = [600, 400]
        self.agent_radius = 15
        self.target_radius = 20
        self.speed = 5
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("DQN Navigation Demo")
        
    def reset(self):
        self.agent_pos = [random.randint(50, self.width-50), 
                         random.randint(50, self.height-50)]
        return self._get_state()
        
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        if action == 0:
            self.agent_pos[1] -= self.speed
        elif action == 1:
            self.agent_pos[0] += self.speed
        elif action == 2:
            self.agent_pos[1] += self.speed
        elif action == 3:
            self.agent_pos[0] -= self.speed
            
        # Keep agent in bounds
        self.agent_pos[0] = max(0, min(self.width, self.agent_pos[0]))
        self.agent_pos[1] = max(0, min(self.height, self.agent_pos[1]))
        
        # Calculate reward based on distance to target
        dist = math.sqrt((self.agent_pos[0] - self.target_pos[0])**2 + 
                        (self.agent_pos[1] - self.target_pos[1])**2)
        reward = -dist/100
        
        # Check if reached target
        done = dist < (self.agent_radius + self.target_radius)
        if done:
            reward = 100
            
        return self._get_state(), reward, done
        
    def _get_state(self):
        # State is [agent_x, agent_y, target_x, target_y]
        return torch.tensor([self.agent_pos[0]/self.width, 
                           self.agent_pos[1]/self.height,
                           self.target_pos[0]/self.width,
                           self.target_pos[1]/self.height], 
                           dtype=torch.float32)
        
    def render(self):
        self.screen.fill((255, 255, 255))
        
        # Draw target
        pygame.draw.circle(self.screen, (255, 0, 0), 
                         (int(self.target_pos[0]), int(self.target_pos[1])), 
                         self.target_radius)
        
        # Draw agent
        pygame.draw.circle(self.screen, (0, 0, 255),
                         (int(self.agent_pos[0]), int(self.agent_pos[1])),
                         self.agent_radius)
                         
        pygame.display.flip()
        
    def close(self):
        pygame.quit()

# Training parameters
state_size = 4  # [agent_x, agent_y, target_x, target_y]
action_size = 4  # up, right, down, left
batch_size = 32
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
num_episodes = 500

# Initialize environment and DQN
env = NavigationEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQNReasoner(state_size, action_size).to(device)
target_dqn = DQNReasoner(state_size, action_size).to(device)
target_dqn.load_state_dict(dqn.state_dict())

optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer()

# Training loop
episode_rewards = []
losses = []
epsilon = epsilon_start

try:
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randrange(action_size)
            else:
                with torch.no_grad():
                    q_values = dqn(state.to(device))
                    action = q_values.argmax().item()
            
            # Take step in environment
            next_state, reward, done = env.step(action)
            
            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Training step
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.stack([s for s in states]).to(device)
                actions = torch.tensor(actions).to(device)
                rewards = torch.tensor(rewards).to(device)
                next_states = torch.stack([s for s in next_states]).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)
                
                current_q_values = dqn(states).gather(1, actions.unsqueeze(1))
                next_q_values = target_dqn(next_states).max(1)[0].detach()
                target_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            state = next_state
            episode_reward += reward
            
            # Render environment
            env.render()
            pygame.time.wait(10)  # Slow down visualization
            
        # Update target network periodically
        if episode % 10 == 0:
            target_dqn.load_state_dict(dqn.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")

except KeyboardInterrupt:
    print("\nTraining interrupted")

finally:
    env.close()

# Plotting results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

# Save the model
torch.save(dqn.state_dict(), 'dqn_reasoner.pth')
