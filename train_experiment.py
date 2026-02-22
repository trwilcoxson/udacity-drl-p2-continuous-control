"""Training script for DDPG Reacher — supports BatchNorm ablation experiment.

Usage:
    python -u train_experiment.py --batch-norm    # baseline (with BatchNorm)
    python -u train_experiment.py --no-batch-norm # ablation (without BatchNorm)

Produces: {prefix}_scores.npy, {prefix}_losses.npz, {prefix}_checkpoint_actor.pth,
          {prefix}_checkpoint_critic.pth
"""
import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from unityagents import UnityEnvironment
from ddpg_agent import Agent

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--batch-norm', dest='use_bn', action='store_true')
group.add_argument('--no-batch-norm', dest='use_bn', action='store_false')
args = parser.parse_args()

prefix = 'bn' if args.use_bn else 'no_bn'
print(f'=== Experiment: {"With" if args.use_bn else "Without"} BatchNorm ===', flush=True)

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
              num_agents=num_agents, random_seed=42,
              use_batch_norm=args.use_bn)

# --- Training loop ---
n_episodes = 300
max_t = 1000
all_scores = []
all_actor_losses = []
all_critic_losses = []
scores_window = deque(maxlen=100)
solved = False
t_start = time.time()

for i_episode in range(1, n_episodes + 1):
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    agent.reset()
    scores = np.zeros(num_agents)

    # Clear per-episode loss accumulators
    agent.actor_losses.clear()
    agent.critic_losses.clear()

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

    # Track episode-averaged losses
    ep_actor_loss = np.mean(agent.actor_losses) if agent.actor_losses else 0.0
    ep_critic_loss = np.mean(agent.critic_losses) if agent.critic_losses else 0.0
    all_actor_losses.append(ep_actor_loss)
    all_critic_losses.append(ep_critic_loss)

    print(f'Episode {i_episode}\tAvg: {rolling_avg:.2f}\tScore: {avg_score:.2f}\t'
          f'ALoss: {ep_actor_loss:.4f}\tCLoss: {ep_critic_loss:.4f}\t({elapsed:.0f}s)',
          flush=True)

    if rolling_avg >= 30.0 and not solved:
        print(f'\n*** Solved at episode {i_episode}!  '
              f'100-Episode Average: {rolling_avg:.2f} ***', flush=True)
        torch.save(agent.actor_local.state_dict(),
                   f'{prefix}_checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(),
                   f'{prefix}_checkpoint_critic.pth')
        solved = True

if not solved:
    print(f'\nDid not solve. Final 100-ep avg: {np.mean(scores_window):.2f}',
          flush=True)
    torch.save(agent.actor_local.state_dict(),
               f'{prefix}_checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(),
               f'{prefix}_checkpoint_critic.pth')

# Save scores and losses
np.save(f'{prefix}_scores.npy', np.array(all_scores))
np.savez(f'{prefix}_losses.npz',
         actor_losses=np.array(all_actor_losses),
         critic_losses=np.array(all_critic_losses))
print(f'Saved {prefix}_scores.npy and {prefix}_losses.npz', flush=True)

# --- Greedy test ---
print('\nRunning 100 greedy test episodes...', flush=True)
agent.actor_local.load_state_dict(
    torch.load(f'{prefix}_checkpoint_actor.pth', weights_only=True))
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

np.save(f'{prefix}_test_scores.npy', np.array(test_scores))

env.close()
print('\nDone!', flush=True)
