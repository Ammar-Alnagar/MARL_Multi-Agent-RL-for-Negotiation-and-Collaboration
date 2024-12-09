import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from typing import List, Tuple

class NegotiationEnvironment:
    """
    A custom environment simulating a negotiation scenario between multiple agents
    """
    def __init__(self, num_agents: int = 2, resource_types: int = 3):
        self.num_agents = num_agents
        self.resource_types = resource_types
        self.reset()
    
    def reset(self):
        """
        Reset the environment to initial state
        """
        # Initial resource distribution
        self.resources = np.random.randint(1, 10, size=(self.num_agents, self.resource_types))
        
        # Collaboration goal: total resources needed
        self.goal_resources = np.random.randint(15, 30, size=self.resource_types)
        
        # Track negotiation steps
        self.current_step = 0
        self.max_steps = 50
        
        return self.resources.copy()
    
    def step(self, actions: List[np.ndarray]) -> Tuple[np.ndarray, float, bool]:
        """
        Process agent actions and return next state, rewards, and done flag
        
        Args:
            actions: List of actions from each agent
        
        Returns:
            next_state: Updated resource distribution
            rewards: Rewards for each agent
            done: Whether episode is complete
        """
        # Apply actions (trade/redistribute resources)
        for i in range(self.num_agents):
            self.resources[i] += actions[i]
        
        # Check collaboration success
        total_resources = np.sum(self.resources, axis=0)
        collaboration_score = np.sum(total_resources >= self.goal_resources)
        
        # Calculate individual rewards
        rewards = [
            collaboration_score / self.resources_types - 
            np.sum(np.abs(actions[i])) * 0.1  # Small penalty for complex trades
            for i in range(self.num_agents)
        ]
        
        self.current_step += 1
        done = (self.current_step >= self.max_steps) or (collaboration_score == self.resources_types)
        
        return self.resources.copy(), rewards, done

class DQNAgent:
    """
    Deep Q-Network Agent for negotiation strategies
    """
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        
        # Neural network for Q-value approximation
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = []
        self.batch_size = 32
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def choose_action(self, state):
        """
        Epsilon-greedy action selection
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """
        Experience replay for learning
        """
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state)
                target = reward + self.gamma * torch.max(self.model(next_state_tensor))
            
            state_tensor = torch.FloatTensor(state)
            target_f = self.model(state_tensor)
            target_f[action] = target
            
            loss = F.mse_loss(self.model(state_tensor), target_f)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class MultiAgentNegotiationTrainer:
    """
    Trainer for multi-agent negotiation using Deep Q-Learning
    """
    def __init__(self, num_agents: int = 2, resource_types: int = 3):
        self.env = NegotiationEnvironment(num_agents, resource_types)
        self.agents = [
            DQNAgent(
                state_size=num_agents * resource_types, 
                action_size=resource_types
            ) for _ in range(num_agents)
        ]
    
    def train(self, episodes: int = 1000):
        """
        Train agents through multiple episodes
        """
        for episode in range(episodes):
            states = self.env.reset()
            done = False
            
            while not done:
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.choose_action(states.flatten())
                    actions.append(np.zeros(self.env.resource_types))
                    actions[i][action] = 1  # Simple action representation
                
                next_states, rewards, done = self.env.step(actions)
                
                # Store experiences
                for i, agent in enumerate(self.agents):
                    agent.remember(
                        states.flatten(), 
                        actions[i], 
                        rewards[i], 
                        next_states.flatten(), 
                        done
                    )
                
                # Learning phase
                for agent in self.agents:
                    agent.replay()
                
                states = next_states
            
            # Print progress
            if episode % 100 == 0:
                print(f"Episode {episode}: Exploration Rate = {self.agents[0].epsilon:.2f}")
    
    def evaluate(self, num_eval_episodes: int = 10):
        """
        Evaluate trained agents' performance
        """
        total_success = 0
        for _ in range(num_eval_episodes):
            states = self.env.reset()
            done = False
            
            while not done:
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.choose_action(states.flatten())
                    actions.append(np.zeros(self.env.resource_types))
                    actions[i][action] = 1
                
                states, rewards, done = self.env.step(actions)
            
            # Check if goal was achieved
            if np.all(np.sum(states, axis=0) >= self.env.goal_resources):
                total_success += 1
        
        success_rate = total_success / num_eval_episodes
        print(f"Evaluation Success Rate: {success_rate * 100:.2f}%")

def main():
    # Create and train multi-agent system
    trainer = MultiAgentNegotiationTrainer(num_agents=3, resource_types=4)
    trainer.train(episodes=2000)
    trainer.evaluate()

if __name__ == "__main__":
    main()