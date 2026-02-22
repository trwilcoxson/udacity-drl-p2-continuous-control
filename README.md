# Continuous Control — Deep Reinforcement Learning

**Author**: Tim Wilcoxson

A DDPG (Deep Deterministic Policy Gradients) agent that learns to control 20 double-jointed arms to track moving target locations in a Unity ML-Agents environment.

## Environment

- **State space**: 33 dimensions per agent (position, rotation, velocity, angular velocity of the arm)
- **Action space**: 4 continuous actions per agent (torque applied to two joints), each in [-1, 1]
- **Agents**: 20 parallel agents sharing a replay buffer
- **Solve condition**: Average score >= 30 over 100 consecutive episodes (averaged across all 20 agents)

## Project Structure

| File | Description |
|---|---|
| `Continuous_Control.ipynb` | Main training notebook with results and experimental comparison |
| `model.py` | Actor and Critic network architectures (configurable BatchNorm) |
| `ddpg_agent.py` | DDPG agent with multi-agent support, OU noise, replay buffer |
| `train_experiment.py` | Standalone training script for BatchNorm ablation experiment |
| `Report.md` | Detailed report: algorithm, results, experimental comparison, future work |
| `checkpoint_actor.pth` | Trained actor weights (baseline with BatchNorm) |
| `checkpoint_critic.pth` | Trained critic weights (baseline with BatchNorm) |
| `scores_plot.png` | Training rewards plot |
| `experiment_comparison.png` | BatchNorm vs. No-BatchNorm comparison plot |
| `loss_curves.png` | Actor and critic loss curves for both configurations |
| `python/` | Bundled Unity ML-Agents Python package (v0.4) |
| `Reacher.app/` | macOS Unity environment (20-agent version) |

## Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- macOS (this project uses the macOS Reacher.app environment)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/trwilcoxson/udacity-drl-p2-continuous-control.git
   cd udacity-drl-p2-continuous-control
   ```

2. Create and activate the conda environment:
   ```bash
   conda create -n drlnd-nav python=3.10 -y
   conda activate drlnd-nav
   ```

3. Install dependencies (includes PyTorch, NumPy, Jupyter, and all other required packages):
   ```bash
   cd python
   pip install .
   cd ..
   ```

4. Install the Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name drlnd-nav --display-name "Python (drlnd-nav)"
   ```

5. Download the Reacher environment (20-agent version) for your OS and place it in the project root:
   - [macOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
   - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
   - [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

   **macOS users**: After unzipping, remove the quarantine attribute so the app can launch:
   ```bash
   xattr -cr Reacher.app
   ```

   **Linux/Windows users**: Update the `file_name` path in the notebook's environment initialization cell to match your downloaded binary.

## Training

```bash
conda activate drlnd-nav
jupyter notebook Continuous_Control.ipynb
```

Select the **"Python (drlnd-nav)"** kernel and run all cells. The agent typically solves the environment in 100–200 episodes.

## Results

See [Report.md](Report.md) for the full learning algorithm description, architecture details, training plot, and ideas for future work.
