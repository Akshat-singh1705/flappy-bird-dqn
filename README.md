# Flappy Bird Deep Q-Network (DQN)

A PyTorch implementation of a Deep Q-Network (DQN) reinforcement learning agent that learns to play Flappy Bird using the `flappy_bird_gymnasium` environment.

This project includes target network synchronization, experience replay, configurable YAML hyperparameters, hardware acceleration support (CUDA / Apple Silicon MPS / CPU), and a manual PyGame interface for human play.

---

## Key Features

* **Deep Q-Network (DQN):** Fully connected feedforward neural network taking 12 state inputs (e.g., bird position, distance to pipes) and predicting Q-values for 2 discrete actions (Flap / Do Nothing).
* **Experience Replay Memory:** Stores transition tuples `(state, action, next_state, reward, done)` in a FIFO queue to decouple consecutive steps and stabilize mini-batch gradient updates.
* **Target Network Synchronization:** Uses a separate target network updated periodically to reduce Q-value overestimation during learning.
* **YAML Configuration:** All training hyperparameter sets are structured cleanly in `parameters.yaml`.
* **Hardware Acceleration:** Automatic hardware selection order: **MPS** (Apple Silicon) -> **CUDA** (NVIDIA GPU) -> **CPU**.
* **Human Play Mode:** A dedicated PyGame script to test and play the environment manually.

---

## Project Structure

```text
├── agent.py               # Main agent script (handles training loop, evaluation, logging)
├── dqn.py                 # PyTorch neural network model architecture
├── experience_replay.py   # Replay memory buffer implementation
├── game_flappy_bird.py    # Manual playable script using PyGame
├── parameters.yaml        # Configuration file for hyperparameters
└── runs/                  # Auto-created directory for model weights (.pt) and logs (.log)
```

---

## Prerequisites & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/flappy-bird-dqn.git
cd flappy-bird-dqn
```

### 2. Install Required Packages
Make sure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install torch gymnasium flappy_bird_gymnasium pygame pyyaml
```

---

## Usage

### 1. Manual Play (Play it Yourself)
To test the environment manually using PyGame controls:
```bash
python game_flappy_bird.py
```
* **Controls:** Press `SPACE` to flap, or do nothing to let gravity drop the bird.

### 2. Train the RL Agent
To start training the agent using the `flappybirdv0` configuration from `parameters.yaml`:
```bash
python agent.py flappybirdv0 --train
```
During training:
* The agent uses an $\epsilon$-greedy strategy, decaying $\epsilon$ after each episode.
* Model checkpoints (`.pt`) and performance logs (`.log`) are automatically saved to the `runs/` directory whenever a new high score is achieved.

### 3. Evaluate a Trained Agent
To visualize the trained agent's behavior (runs greedily with $\epsilon = 0.0$ and renders the game window):
```bash
python agent.py flappybirdv0
```

---

## Hyperparameters

Training parameters are stored in `parameters.yaml`. You can modify existing values or add new experiment profiles:

```yaml
flappybirdv0:
  env_id: FlappyBird-v0
  epsilon_init: 1.0         # Initial exploration rate
  epsilon_min: 0.05         # Minimum exploration rate
  epsilon_decay: 0.9995     # Decay factor per episode
  replay_memory_size: 100000 # Max experience buffer capacity
  mini_batch_size: 32       # Batch size for Q-network optimization
  network_syn_rate: 10      # Target network sync frequency (in steps)
  alpha: 0.001              # Learning rate for Adam optimizer
  gamma: 0.99               # Discount factor for future rewards
  reward_threshold: 1000    # Target max reward per episode
```

---

## How It Works

1. **State Space:** 12-dimensional vector providing relative positions and velocities of the bird and nearby pipe obstacles.
2. **Action Space:** Discrete space with 2 actions:
   * `0`: Do nothing
   * `1`: Flap wings
3. **Loss Function:** Mean Squared Error (MSE) computed between current policy Q-values and target Q-values computed via the Bellman equation:
   $$Q_{\text{target}}(s, a) = r + \gamma \cdot \max_{a'} Q_{\text{target\_net}}(s', a')$$

---
Thanks for Visiting.
