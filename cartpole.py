import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, seed=42):
        """Initialize a Deep Q-Learning Agent for the CartPole environment.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Number of possible actions
            seed (int): Random seed for reproducibility
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural Network for Q-value approximation
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=1e-3)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, 10000, seed)
        
        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.tau = 1e-3    # Soft update of target parameters
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        """Store experience in replay memory and learn periodically."""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step += 1
        if self.t_step % 4 == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
        
    def act(self, state, eps=None):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Set epsilon for action selection
        eps = self.eps_start if eps is None else eps
        
        # Turn off gradient computation for inference
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.qnetwork_local.action_size))
        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Number of actions
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        """Build a network that maps states -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, seed):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = 64
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def train_dqn(env_name='CartPole-v1', n_episodes=1000, max_t=1000):
    """Deep Q-Learning training function for CartPole environment.
    
    Params:
    ======
        env_name (str): Name of the Gym environment
        n_episodes (int): Maximum number of training episodes
        max_t (int): Maximum number of timesteps per episode
    """
    # Initialize environment and agent
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    # Track performance metrics
    scores = []
    scores_window = deque(maxlen=100)
    eps = agent.eps_start
    
    # Training loop
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Agent learns from experience
            agent.step(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            if done:
                break
        
        # Update epsilon for exploration
        eps = max(agent.eps_end, agent.eps_decay * eps)
        
        # Record performance
        scores_window.append(score)
        scores.append(score)
        
        # Print progress
        if i_episode % 100 == 0:
            print(f'Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        
        # Check if environment is solved
        if np.mean(scores_window) >= 195.0:
            print(f'\nEnvironment solved in {i_episode-100} episodes!')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    
    return scores

# Main execution
if __name__ == "__main__":
    # Train the DQN agent
    scores = train_dqn()
    
    # Optional: Plot learning curve or visualize results
    import matplotlib.pyplot as plt
    plt.plot(scores)
    plt.title('DQN Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()