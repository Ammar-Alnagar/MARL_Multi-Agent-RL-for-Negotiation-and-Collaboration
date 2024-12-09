import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from typing import List, Tuple, Dict

class NegotiationEnvironment:
    def __init__(self, num_agents: int = 3):
        """
        Simulation environment for multi-agent negotiation
        
        Args:
            num_agents (int): Number of agents in the negotiation
        """
        self.num_agents = num_agents
        self.resources = {
            'money': 1000,
            'time': 100,
            'expertise': 50
        }
        self.negotiation_rounds = 0
        self.max_rounds = 10
        
    def reset(self):
        """Reset the environment for a new negotiation"""
        self.negotiation_rounds = 0
        self.resources = {
            'money': 1000,
            'time': 100,
            'expertise': 50
        }
        return self._get_state()
    
    def _get_state(self) -> Dict[str, float]:
        """Generate current state representation"""
        return {
            'resources': self.resources,
            'round': self.negotiation_rounds
        }
    
    def step(self, actions: List[Dict[str, float]]) -> Tuple[Dict, List[float], bool, Dict]:
        """
        Process a negotiation step
        
        Args:
            actions (List[Dict]): Actions proposed by each agent
        
        Returns:
            next_state, rewards, done, info
        """
        # Validate and process actions
        valid_actions = self._validate_actions(actions)
        
        # Calculate collective utility
        collective_utility = self._calculate_collective_utility(valid_actions)
        
        # Calculate individual rewards
        rewards = self._calculate_rewards(valid_actions, collective_utility)
        
        # Update resources
        self._update_resources(valid_actions)
        
        self.negotiation_rounds += 1
        done = self.negotiation_rounds >= self.max_rounds
        
        return (
            self._get_state(), 
            rewards, 
            done, 
            {'actions': valid_actions}
        )
    
    def _validate_actions(self, actions: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Validate and constrain agent actions"""
        validated_actions = []
        for action in actions:
            validated_action = {
                'money_offer': max(0, min(action.get('money_offer', 0), self.resources['money'])),
                'time_commitment': max(0, min(action.get('time_commitment', 0), self.resources['time'])),
                'expertise_share': max(0, min(action.get('expertise_share', 0), self.resources['expertise']))
            }
            validated_actions.append(validated_action)
        return validated_actions
    
    def _calculate_collective_utility(self, actions: List[Dict[str, float]]) -> float:
        """Compute overall negotiation utility"""
        total_money = sum(action['money_offer'] for action in actions)
        total_time = sum(action['time_commitment'] for action in actions)
        total_expertise = sum(action['expertise_share'] for action in actions)
        
        return (total_money * 0.4) + (total_time * 0.3) + (total_expertise * 0.3)
    
    def _calculate_rewards(self, actions: List[Dict[str, float]], collective_utility: float) -> List[float]:
        """Calculate individual agent rewards"""
        rewards = []
        for action in actions:
            individual_utility = (
                action['money_offer'] * 0.4 + 
                action['time_commitment'] * 0.3 + 
                action['expertise_share'] * 0.3
            )
            reward = individual_utility / collective_utility if collective_utility > 0 else 0
            rewards.append(reward)
        return rewards
    
    def _update_resources(self, actions: List[Dict[str, float]]):
        """Update environment resources based on agent actions"""
        for action in actions:
            self.resources['money'] -= action['money_offer']
            self.resources['time'] -= action['time_commitment']
            self.resources['expertise'] -= action['expertise_share']

class NegotiationAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        """
        Deep Q-Network for negotiation agent
        
        Args:
            state_dim (int): Dimension of state representation
            action_dim (int): Dimension of action space
        """
        super(NegotiationAgent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, state):
        """Forward pass through neural network"""
        return self.network(state)

class MultiAgentNegotiationTrainer:
    def __init__(self, num_agents: int = 3, state_dim: int = 10, action_dim: int = 3):
        """
        Multi-agent training framework
        
        Args:
            num_agents (int): Number of negotiation agents
            state_dim (int): State representation dimension
            action_dim (int): Action space dimension
        """
        self.environment = NegotiationEnvironment(num_agents)
        self.agents = [NegotiationAgent(state_dim, action_dim) for _ in range(num_agents)]
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.01
        self.gamma = 0.95  # Discount factor
    
    def train(self, num_episodes: int = 1000):
        """
        Train multi-agent system through episodic learning
        
        Args:
            num_episodes (int): Number of training episodes
        """
        for episode in range(num_episodes):
            state = self.environment.reset()
            done = False
            
            while not done:
                # Collect actions from all agents
                actions = []
                for agent in self.agents:
                    action = self._select_action(agent, state)
                    actions.append(action)
                
                # Environment step
                next_state, rewards, done, _ = self.environment.step(actions)
                
                # Update agents
                for i, (agent, reward) in enumerate(zip(self.agents, rewards)):
                    self._update_agent(agent, state, actions[i], reward, next_state, done)
                
                state = next_state
            
            # Decay exploration rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def _select_action(self, agent, state):
        """
        Epsilon-greedy action selection
        
        Args:
            agent (NegotiationAgent): Agent selecting action
            state (Dict): Current environment state
        
        Returns:
            Dict: Selected negotiation action
        """
        if random.random() < self.epsilon:
            return {
                'money_offer': random.uniform(0, 100),
                'time_commitment': random.uniform(0, 10),
                'expertise_share': random.uniform(0, 5)
            }
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(list(state.values()))
            q_values = agent(state_tensor)
            action_index = torch.argmax(q_values).item()
            
            # Map index to action space
            return {
                'money_offer': q_values[0].item(),
                'time_commitment': q_values[1].item(),
                'expertise_share': q_values[2].item()
            }
    
    def _update_agent(self, agent, state, action, reward, next_state, done):
        """
        Update agent's Q-network using experience replay
        
        Args:
            agent (NegotiationAgent): Agent to update
            state (Dict): Current state
            action (Dict): Taken action
            reward (float): Received reward
            next_state (Dict): Next state
            done (bool): Episode termination flag
        """
        state_tensor = torch.FloatTensor(list(state.values()))
        next_state_tensor = torch.FloatTensor(list(next_state.values()))
        
        # Compute current Q-values
        current_q = agent(state_tensor)
        
        # Compute target Q-values
        target_q = current_q.clone().detach()
        if done:
            target_q = torch.tensor([reward])
        else:
            target_q = reward + self.gamma * torch.max(agent(next_state_tensor))
        
        # Compute loss and update
        loss = agent.loss_fn(current_q, target_q)
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

def main():
    """Main training execution"""
    trainer = MultiAgentNegotiationTrainer(num_agents=3)
    trainer.train(num_episodes=5000)
    print("Multi-Agent Negotiation Training Completed!")

if __name__ == "__main__":
    main()