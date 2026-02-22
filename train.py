"""Standalone training script for DDPG Reacher (20 agents).

Run: conda activate drlnd-nav && python -u train.py
Produces: checkpoint_actor.pth, checkpoint_critic.pth, scores.npy, scores_plot.png
"""
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from collections import deque
from unityagents import UnityEnvironment
from ddpg_agent import Agent

# --- Environment setup ---
env = UnityEnvironment(file_name='Reacher.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state_size = env_info.vector_observations.shape[1]

print(f'Agents: {num_agents}  State: {state_size}  Action: {action_size}', flush=True)

# --- Agent ---
agent = Agent(state_size=state_size, action_size=action_size,
              num_agents=num_agents, random_seed=42)

# --- Training loop ---
n_episodes = 300
max_t = 1000
all_scores = []
scores_window = deque(maxlen=100)
solved = False
t_start = time.time()

for i_episode in range(1, n_episodes + 1):
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    agent.reset()
    scores = np.zeros(num_agents)

    for t in range(max_t):
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        agent.step(states, actions, rewards, next_states, dones, t)
        states = next_states
        scores += rewards
        if np.any(dones):
            break

    avg_score = np.mean(scores)
    all_scores.append(avg_score)
    scores_window.append(avg_score)
    rolling_avg = np.mean(scores_window)
    elapsed = time.time() - t_start

    # Print every episode on its own line for log file readability
    print(f'Episode {i_episode}\tAvg: {rolling_avg:.2f}\tScore: {avg_score:.2f}\t({elapsed:.0f}s)', flush=True)

    if rolling_avg >= 30.0 and not solved:
        print(f'\n*** Environment solved in {i_episode - 100} episodes!  Average Score: {rolling_avg:.2f} ***', flush=True)
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        solved = True

if not solved:
    print(f'\nDid not solve. Final 100-ep avg: {np.mean(scores_window):.2f}', flush=True)
    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

# Save raw scores for notebook
np.save('scores.npy', np.array(all_scores))

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(np.arange(1, len(all_scores) + 1), all_scores, alpha=0.3, color='steelblue', label='Episode Score')
if len(all_scores) >= 100:
    rolling = [np.mean(all_scores[max(0, i-100):i]) for i in range(1, len(all_scores) + 1)]
    ax.plot(np.arange(1, len(all_scores) + 1), rolling, color='darkblue', linewidth=2, label='100-Episode Average')
ax.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Solve Threshold (30)')
ax.set_xlabel('Episode')
ax.set_ylabel('Average Score (20 Agents)')
ax.set_title('DDPG Training — Continuous Control (20 Agents)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scores_plot.png', dpi=150, bbox_inches='tight')
print('Saved scores_plot.png', flush=True)

# --- Greedy test ---
print('\nRunning 100 greedy test episodes...', flush=True)
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth', weights_only=True))
test_scores = []
for i in range(1, 101):
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    ep_scores = np.zeros(num_agents)
    while True:
        actions = agent.act(states, add_noise=False)
        env_info = env.step(actions)[brain_name]
        states = env_info.vector_observations
        ep_scores += env_info.rewards
        if np.any(env_info.local_done):
            break
    avg = np.mean(ep_scores)
    test_scores.append(avg)
    if i % 10 == 0:
        print(f'Test {i}/100  Score: {avg:.2f}', flush=True)

print(f'\nGreedy Test Results (100 episodes):', flush=True)
print(f'  Average: {np.mean(test_scores):.2f}', flush=True)
print(f'  Std Dev: {np.std(test_scores):.2f}', flush=True)
print(f'  Min:     {np.min(test_scores):.2f}', flush=True)
print(f'  Max:     {np.max(test_scores):.2f}', flush=True)

env.close()
print('\nDone!')
