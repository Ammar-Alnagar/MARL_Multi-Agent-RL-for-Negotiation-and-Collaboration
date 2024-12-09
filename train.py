env = NegotiationEnv()
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent1 = DDPGAgent(obs_dim, action_dim)
agent2 = DDPGAgent(obs_dim, action_dim)

num_episodes = 500
batch_size = 64
replay_buffer = []

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Agents act
        action1 = agent1.act(obs)
        action2 = agent2.act(obs)
        actions = [action1[0], action2[0]]

        # Environment step
        next_obs, rewards, done, _ = env.step(actions)
        replay_buffer.append((obs, actions, rewards, next_obs, done))
        if len(replay_buffer) > batch_size:
            replay_buffer.pop(0)
        
        obs = next_obs
        episode_reward += sum(rewards)
        
        # Update agents with sampled transitions
        if len(replay_buffer) >= batch_size:
            transitions = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*transitions)
            agent1.update((states, actions, rewards, next_states, dones))
            agent2.update((states, actions, rewards, next_states, dones))

    print(f"Episode {episode + 1}: Total Reward: {episode_reward}")