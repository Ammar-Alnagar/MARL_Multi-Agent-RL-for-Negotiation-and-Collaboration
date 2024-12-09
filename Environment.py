import numpy as np
import gymnasium as gym
from gymnasium import spaces

class NegotiationEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Observation: Each agent observes the current resource pool (total 100 units)
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        # Action: Each agent proposes how much of the resource they want (0 to 50 units)
        self.action_space = spaces.Box(low=0, high=50, shape=(2,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.resource_pool = 100  # Total shared resource
        return np.array([self.resource_pool, self.resource_pool], dtype=np.float32)

    def step(self, actions):
        agent1_request, agent2_request = actions
        total_request = agent1_request + agent2_request

        # Reward logic: Collaboration works better than competition
        if total_request <= self.resource_pool:
            reward = (agent1_request + agent2_request) / 2  # Shared reward
            self.resource_pool -= total_request
        else:
            reward = -1  # Penalty for over-requesting

        # Observations after action
        obs = np.array([self.resource_pool, self.resource_pool], dtype=np.float32)
        done = self.resource_pool <= 0  # Episode ends when resources are depleted
        return obs, [reward, reward], done, {}

    def render(self):
        print(f"Resource pool: {self.resource_pool}")