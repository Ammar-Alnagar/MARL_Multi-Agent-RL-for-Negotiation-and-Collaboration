import torch
import torch.nn as nn
import torch.optim as optim
import random

class DDPGAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3):
        self.policy_net = self._build_network(obs_dim, action_dim)
        self.target_policy_net = self._build_network(obs_dim, action_dim)
        self.critic_net = self._build_network(obs_dim + action_dim, 1)
        self.target_critic_net = self._build_network(obs_dim + action_dim, 1)
        
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr=lr)
        
        self.gamma = 0.99
        self.tau = 0.005  # Soft update parameter

    def _build_network(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def act(self, state, noise=0.1):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.policy_net(state).detach().numpy()
        return np.clip(action + noise * np.random.randn(*action.shape), 0, 50)

    def update(self, transitions):
        states, actions, rewards, next_states, dones = transitions
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Update critic
        with torch.no_grad():
            target_actions = self.target_policy_net(next_states)
            target_q_values = rewards + self.gamma * self.target_critic_net(torch.cat((next_states, target_actions), dim=1)) * (1 - dones)
        q_values = self.critic_net(torch.cat((states, actions), dim=1))
        critic_loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update policy
        policy_loss = -self.critic_net(torch.cat((states, self.policy_net(states)), dim=1)).mean()
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        # Soft update of target networks
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)