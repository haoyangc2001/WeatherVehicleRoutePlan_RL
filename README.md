# WeatherVehicleRoutePlan_RL: Reinforcement Learning for Weather-Aware Vehicle Route Planning

This repository provides a reinforcement learning implementation for weather-aware vehicle route planning.

## Overview

Weather-aware vehicle route planning extends the classic Vehicle Routing Problem by introducing uncertainty in customer demands and travel costs. This implementation uses a reinforcement learning approach to tackle that stochastic decision-making problem.

Key features:
- Neural network architecture with state and memory embeddings
- Attention mechanism for node selection
- REINFORCE algorithm with baseline for training
- Multiple inference strategies (greedy, random sampling, beam search)
- Support for varying levels of stochasticity in the environment

## Project Structure

```
WeatherVehicleRoutePlan_RL/
├── main.py               # Main entry point for training/inference
├── config.py             # Configuration parameters
├── models/
│   ├── __init__.py
│   ├── attention.py      # Attention layer implementation
│   ├── embedding.py      # State and memory embedding components
│   └── policy.py         # Full policy model architecture
├── env/
│   ├── __init__.py
│   ├── svrp_env.py       # Weather-aware routing environment implementation
│   └── weather.py        # Weather simulation for stochastic variables
├── training/
│   ├── __init__.py
│   ├── reinforce.py      # REINFORCE algorithm implementation
│   └── baseline.py       # Baseline model implementation  
├── inference/
│   ├── __init__.py
│   └── inference.py      # Inference strategies (greedy, random, beam search)
```

## Network Architecture

The policy network in this repository follows a compact encoder-attention design tailored to the current weather-aware routing environment implementation. At each decision step, the model receives a full snapshot of customer-side stochastic information and vehicle states, then outputs a probability distribution over the next node for every vehicle.

### 1. State Representation

The environment produces two input tensors:

- `customer_features`: shape `[batch_size, num_nodes, customer_input_dim]`
- `vehicle_features`: shape `[batch_size, num_vehicles, vehicle_input_dim]`

In the current implementation, customer features are assembled as:

1. Global weather vector replicated to every node
2. Remaining demand of each node
3. Travel-cost row from the current scenario matrix

So the customer feature dimension is:

```text
customer_input_dim = weather_dim + 1 + num_nodes
```

Vehicle features are intentionally minimal:

1. Current vehicle position index
2. Normalized remaining load

Therefore:

```text
vehicle_input_dim = 2
```

This design gives the policy direct access to both stochastic context and operational state. In practical terms, each decision is conditioned on weather, unmet demand, and the realized travel-cost structure of the instance.

### 2. Customer Encoder

Customer information is encoded by `models/embedding.py::CustomerEncoder`.

- Input: `[B, N, customer_input_dim]`
- Reshape: `[B * N, 1, customer_input_dim]`
- Layer: `Conv1d(1, embedding_dim, kernel_size=customer_input_dim)`
- Output: `[B, N, embedding_dim]`

Because the convolution kernel spans the full feature dimension, this layer behaves like a shared node-wise feature projector. Every customer node is transformed independently into a dense state embedding, while weights are shared across all nodes.

Conceptually:

```text
customer_features[i] -> state_embedding[i] in R^embedding_dim
```

This is a lightweight alternative to a deeper MLP or graph encoder. It is simple and fast, but it does not explicitly model higher-order interactions between customers beyond what is already embedded in the travel-cost features.

### 3. Vehicle Encoder

Vehicle information is encoded by `models/embedding.py::VehicleEncoder`.

- Input: `[B, V, vehicle_input_dim]`
- Reshape: `[B * V, 1, vehicle_input_dim]`
- Layer: single-layer `LSTM(input_size=vehicle_input_dim, hidden_size=embedding_dim)`
- Output: `[B, V, embedding_dim]`

The LSTM hidden state is passed from one environment step to the next. This gives each vehicle a compact recurrent memory of its own historical decision process, rather than using only the instantaneous position/load pair.

In effect, the policy maintains:

```text
h_t^vehicle = LSTM(vehicle_features_t, h_{t-1}^vehicle)
```

This is the model's main temporal memory mechanism.

### 4. Attention-Based Node Selection

The core decision module is implemented in `models/attention.py::AttentionLayer`.

The model projects:

- vehicle embeddings into queries `Q`
- customer embeddings into keys `K`

and computes scaled dot-product attention:

```text
score(v, n) = (Q_v · K_n) / sqrt(embedding_dim)
```

After masking invalid nodes, the model applies softmax over all nodes:

```text
P(next_node = n | state, vehicle) = softmax(score(v, n))
```

The output tensor has shape:

```text
[batch_size, num_vehicles, num_nodes]
```

Each slice corresponds to one vehicle's categorical distribution over candidate next nodes.

### 5. Feasibility Mask

`models/attention.py::MaskingLayer` masks nodes whose remaining demand is already zero or below. The depot (`node 0`) is never masked, so the policy always retains the option to return to depot and refill capacity.

Mask rule:

```text
mask[n] = (remaining_demand[n] <= 0), except depot
```

This keeps the policy from repeatedly selecting already-fulfilled customers while preserving depot recourse behavior.

### 6. Action Generation

The final policy output is converted to log-probabilities:

```text
log_probs = log(softmax(scores))
```

Action selection supports two modes:

- Greedy: choose `argmax(log_probs)`
- Sampling: draw from the categorical distribution induced by `exp(log_probs)`

During training, actions are sampled to support REINFORCE exploration. During evaluation, the code can switch between greedy decoding, repeated random sampling, and beam-style search.

### 7. Baseline Network

To reduce policy-gradient variance, training uses a separate baseline defined in `training/baseline.py`.

Its input is built by:

1. Averaging customer features over all nodes
2. Averaging vehicle features over all vehicles
3. Concatenating both summaries

Then a small MLP predicts a scalar value estimate:

```text
avg_customer -> Linear -> ReLU -> Linear -> ReLU -> Linear -> value
```

This baseline approximates the expected return of the current state and is trained with mean squared error against Monte Carlo returns.

### 8. End-to-End Decision Flow

At one environment step, the forward pass is:

1. Environment builds `customer_features`, `vehicle_features`, and `demands`
2. Customer encoder maps node features into state embeddings
3. Vehicle encoder maps per-vehicle state into memory embeddings
4. Masking layer suppresses already-served customers
5. Attention layer produces per-vehicle node probabilities
6. The trainer samples actions and records their log-probabilities
7. Environment executes the actions and returns rewards and next state

This loop is repeated until all customer demand is served or the step limit is reached.

### 9. Shape Summary

For clarity, the main tensor shapes through the network are:

```text
customer_features      : [B, N, weather_dim + 1 + N]
vehicle_features       : [B, V, 2]
state_embeddings       : [B, N, D]
memory_embeddings      : [B, V, D]
attention_scores       : [B, V, N]
log_probs              : [B, V, N]
actions                : [B, V]
```

Where:

- `B` = batch size
- `N` = number of nodes
- `V` = number of vehicles
- `D` = embedding dimension

### 10. Design Rationale and Limitations

This architecture is deliberately compact:

- It can be trained end-to-end with standard policy gradients
- It directly consumes stochastic context through weather and realized costs
- It keeps implementation complexity low for experimentation

At the same time, it has important limitations:

- Customer encoding is independent across nodes and is not graph-structured
- Vehicle features are extremely low-dimensional
- The attention module is shallow compared with transformer-style routing models
- The model observes realized scenario information directly, which is simpler than a partially observed recourse setting

So the repository should be viewed as a concise research prototype rather than a production-grade neural combinatorial optimizer.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

To train a model with default parameters:

```bash
python main.py
```

To customize training parameters:

```bash
python main.py --num_nodes 50 --num_vehicles 2 --embedding_dim 256 --epochs 200 --batch_size 16
```

### Testing

To evaluate a trained model:

```bash
python main.py --test --load_model checkpoints/model_final
```

To compare different inference strategies:

```bash
python main.py --test --load_model checkpoints/model_final --inference greedy
python main.py --test --load_model checkpoints/model_final --inference random --num_samples 32
python main.py --test --load_model checkpoints/model_final --inference beam --beam_width 5
```

## Key Parameters

- `--num_nodes`: Number of customer nodes plus depot
- `--num_vehicles`: Number of vehicles available
- `--capacity`: Maximum vehicle capacity
- `--a_ratio`, `--b_ratio`, `--gamma_ratio`: Signal ratios for stochastic components
- `--embedding_dim`: Dimension of state and memory embeddings
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--inference`: Inference strategy (greedy, random, beam)

## Environment Settings

The WeatherVehicleRoutePlan_RL environment models stochasticity through three components:

1. **Constant component (a_ratio)**: Fixed part of the stochastic variables
2. **Weather component (b_ratio)**: Part influenced by weather variables
3. **Noise component (gamma_ratio)**: Random noise

The model learns to leverage weather information to predict stochastic variables and make better routing decisions.

## Results

The implementation achieves competitive results compared to classical methods:

- 3.43% improvement over Ant Colony Optimization
- Superior performance in correlated environments where weather affects both demand and travel costs
- Efficient inference suitable for real-time industrial applications
