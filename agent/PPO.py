import torch
import torch.nn as nn

from RL2.agent.Level2_agent import rule_agent
from RL2.environment import *

import  csv
import  matplotlib.pyplot as plt

import pandas as pd

critic=Critic()
actor=Actor()
best=0

actor_optimizer=torch.optim.Adam(actor.parameters())

critic_optimizer=torch.optim.Adam(critic.parameters())

landlord_win=0

for episodes in range(100):

    batch_trace=[]

    for episode in range(500):

        trace=[]

        env = Doudizhu()

        env.reset()

        while not env.done:

            final_action = None

            hand = env.hands[env.current_player]

            if env.current_player == env.landlord:

                scores=[]

                state = torch.tensor(env._get_state(), dtype=torch.float)

                for act in env.fast_all_legal_actions(hand):

                    action = torch.tensor(env.convert_actions(act), dtype=torch.float)

                    q_value=  actor(torch.cat([state,action]))

                    scores.append(q_value)

                scores = torch.stack(scores).squeeze(-1)

                probs = torch.softmax(scores, dim=0)

                dist = torch.distributions.Categorical(probs)

                idx = dist.sample()

                log_prob = dist.log_prob(idx)

                action= env.fast_all_legal_actions(hand)

                final_action = action[idx]

                trace.append((state,idx,env.fast_all_legal_actions(hand),log_prob.detach()))

                env.step(final_action)

            else:

                action = rule_agent(env)

                env.step(action)

        if env.winner == env.landlord:

            reward = 1

            landlord_win+=1

        else:
            reward = -1

        batch_trace.extend((s, i, a, lp, reward) for s, i, a, lp in trace)

    for _ in range(4):

        actor_total_loss = 0

        critic_total_loss = 0

        for state,idx,actions,log_prob,reward in batch_trace:

            value=critic(state)

            advantage = reward - value

            scores = []

            for action in actions:

                action_tensor = torch.tensor(env.convert_actions(action), dtype=torch.float)

                score = actor(torch.cat([state, action_tensor]))

                scores.append(score)

            scores = torch.stack(scores).squeeze(-1)

            probs = torch.softmax(scores, dim=0)

            dist = torch.distributions.Categorical(probs)

            new_log_prob = dist.log_prob(idx)

            ratio = torch.exp(new_log_prob - log_prob)

            clip_ratio = torch.clamp(ratio, min=0.8, max=1.2)

            actor_loss = -torch.min(ratio * advantage.detach(), clip_ratio * advantage.detach())

            critic_loss = (value - reward) ** 2

            actor_total_loss += actor_loss

            critic_total_loss += critic_loss

        actor_total_loss = actor_total_loss / len(batch_trace)
        critic_total_loss = critic_total_loss / len(batch_trace)

        actor_optimizer.zero_grad()

        actor_total_loss.backward()

        actor_optimizer.step()

        critic_optimizer.zero_grad()

        critic_total_loss.backward()

        critic_optimizer.step()

    if (episode+1) % 500 == 0:

        new=landlord_win/500

        print('winning rate:',int(new*100))

        if new>best:

            best=new

            torch.save(actor.state_dict(),'model/ppo.basic.pth')

        landlord_win=0

        with open("graph/ppo_training_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episodes * 500, int(new * 100)])

df = pd.read_csv("graph/ppo_training_log.csv", header=None, names=["episode", "winning_rate"])

plt.plot(df["episode"], df["winning_rate"])
plt.xlabel("Episode")
plt.ylabel("Winning Rate")
plt.title("PPOTraining Win Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig("graph/ppo_basic_training_curve_66state.png")

print('END')



















































