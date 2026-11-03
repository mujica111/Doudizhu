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


### 4.2 Further Reinforcement Learning Experiment Against Rulebase 2


Following the initial experiments, Bellman-based DQN achieved approximately **80% win rate against Rulebase 1**, highlighting the importance of temporal credit assignment. Double DQN was subsequently explored to improve training stability.

However, improvements against a simple opponent eventually became less informative. A stronger benchmark, **Rulebase 2**, was introduced to evaluate whether the learned policies could handle more challenging decision-making scenarios. This opponent achieved approximately 99% win rate against random play.

**Initial Evaluation Against Rulebase 2**

| Method | Win Rate |
|---|---:|
| Bellman-based DQN | 52.5% |
| Monte Carlo | 33.5% |
| Monte Carlo Self-play | 48.4% |

These results revealed substantial differences between the learning approaches. Bellman-based DQN demonstrated stronger initial performance, while Monte Carlo methods struggled to achieve comparable results.

**Further Training and Optimization**

Training directly against Rulebase 2 improved Monte Carlo performance to approximately 47%, while Bellman-based DQN initially reached around 55%. Subsequent improvements to state representation increased Bellman-based DQN performance from approximately 56% to 61%.

PPO was then introduced as an alternative policy optimization approach. Initial experiments achieved approximately 58% win rate, with later configurations reaching around 60% and exhibiting relatively stable learning behavior.

**Figure 2 — State Representation and PPO Hyperparameter Experiments**

![figure2](image/figure2.png)]

**Key Findings**

- **Bellman-based learning:** Incorporating future rewards substantially improved performance against the initial benchmark. Bellman-based DQN also demonstrated strong results against Rulebase 2, with further gains from state representation improvements.
- **Monte Carlo limitations:** Despite additional training and self-play experiments, Monte Carlo methods generally underperformed the stronger Bellman-based configurations.
- **PPO training stability:** PPO achieved competitive performance with relatively stable training behavior. Experiments with different learning rates, clipping ranges, and state representations produced limited additional improvements.
- **Diminishing returns:** Repeated tuning eventually produced smaller performance gains, motivating a transition toward stronger opponents and alternative training approaches.

**Algorithm Selection**

Based on these experiments, **Bellman-based DQN and PPO were selected for further investigation**. Bellman-based DQN demonstrated strong performance improvements, particularly through state representation refinement, while PPO offered a relatively stable policy optimization framework.

The next stage introduced DouZero as a substantially stronger opponent. Initial evaluation of the best PPO agent achieved only approximately **26% win rate**, exposing a significant gap between performance against rule-based opponents and more competitive pretrained agents.

This motivated subsequent experiments with expert demonstrations, behavioral cloning, and reinforcement learning guided by pretrained policies.



