## Overview

A from-scratch exploration of how to effectively train a competitive Dou Dizhu agent. Starting with reinforcement learning from basic baselines, the project progressively explores expert-guided training, leveraging strong open-source agents such as DouZero as behavioral references to improve policy learning and overcome the limitations of training from scratch.

## Core Components

- **Custom Game Environment:** Built a Dou Dizhu environment with efficient legal-action generation through card-pattern matching, significantly improving simulation throughput.
- **Reinforcement Learning:** Explored and compared different training approaches, with iterative improvements to reward design, state representation, and policy optimization.
- **Expert-Guided Learning:** Incorporated strong pretrained agents such as DouZero as expert references, exploring ways to combine expert knowledge with reinforcement learning.
- **DouZero Integration:** Developed an adapter interface connecting the custom game environment with DouZero, enabling pretrained DouZero agents to participate in games while preserving the original agent's decision-making behavior and playing strength as closely as possible.
- **Benchmark & Evaluation:** Developed progressively stronger rule-based benchmarks and a fixed opponent pool to evaluate policy improvements and identify training limitations.

## Training Methodology

### 1. Reinforcement Learning from Scratch
Established initial training baselines using DQN, Monte Carlo methods, and PPO. Experimented with reward design, state representation, and temporal credit assignment to improve learning performance.

### 2. Expert-Guided Policy Learning
Introduced strong pretrained agents such as DouZero as sources of expert demonstrations. Explored combining expert-guided policy learning with reinforcement learning to overcome the limitations of training against weaker or overly strong opponents.

### 3. Opponent Pool and Policy Selection
Developed a fixed opponent pool consisting of diverse strategies, including DouZero and officially released variants from the DouZero team. Introduced repeated evaluation and tournament-based model selection to encourage consistent policy improvement.

## Experimental Results

### 4.1 Reinforcement Learning Against Rulebase 1

Initial experiments focused on training agents from scratch against a simple rule-based opponent that consistently selected the smallest legal action(91% winning percentage vs random choice agent).

DQN, Bellman-based Q-learning, and Monte Carlo methods were compared to investigate their learning behavior and training stability so as to determine suitable training methods. Monte Carlo self-play was subsequently introduced to examine whether agents could improve through interaction with copies of their own policy.

**Figure 1 — Reinforcement Learning Experiments Against Rulebase 1**

![figure1](image/rule_base1.png)

**Key Findings**

- **Learning effectiveness:** Bellman-based Q-learning and Monte Carlo methods achieved substantial improvements over the initial reward-based DQN baseline.
- **Training instability:** Monte Carlo exhibited significant performance fluctuations, even after achieving high win rates.
- **Self-play evaluation:** Monte Carlo self-play achieved a peak evaluation win rate of **78.8%** against Rulebase 1, but subsequent checkpoints showed substantial performance degradation.
- **Model selection:** These experiments highlighted the importance of independent evaluation and best-checkpoint selection rather than relying on the final trained model.


