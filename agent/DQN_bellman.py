import pandas as pd
import torch
import torch.nn as nn

from RL2.agent.Level1_agent import rule1_agent
from RL2.environment import *
import csv

import pandas as pt
import matplotlib.pyplot as plt

net=QNetwork()

target_net=QNetwork()

target_net.load_state_dict(net.state_dict())

optimizer=torch.optim.Adam(net.parameters())

landlord_win=0

best=0

gamma=0.99

for episode in range(90000):

    if episode%600==1:

        target_net.load_state_dict(net.state_dict())

    env=Doudizhu()

    env.reset()

    while not env.done:

        final_action =None

        hand=env.hands[env.current_player]

        if env.current_player==env.landlord:

            best_q = float('-inf')

            for act in env.fast_all_legal_actions(hand):

                action= torch.tensor(env.convert_actions(act),dtype=torch.float)

                state= torch.tensor(env._get_state(),dtype=torch.float)

                q=torch.cat([state,action])

                q_value=net(q)

                q_target = target_net(torch.cat([state, action]))

                if best_q<q_value:

                    best_q=q_value

                    final_action=act

                    q_target_value=q_target.detach()

            if random.random()<0.1 and episode>=2999 :

                final_action=random.choice(env.fast_all_legal_actions(hand))

                action = torch.tensor(env.convert_actions(final_action), dtype=torch.float)

                best_q = net(torch.cat([state, action]))

            reward=0

            if final_action==['pass']:

                reward-=0.1

            else:
                for card in final_action:

                    reward-=int(card)*0.01

            env.step(final_action)

        else:
            env.step(rule1_agent(env))

            if not env.done and env.current_player==env.landlord:

                hand = env.hands[env.current_player]

                best_next_q = float('-inf')

                for act in env.fast_all_legal_actions(hand):

                    action = torch.tensor(env.convert_actions(act), dtype=torch.float)

                    state = torch.tensor(env._get_state(), dtype=torch.float)

                    next_q = net(torch.cat([state, action]))

                    if best_next_q < next_q:

                        best_next_q = next_q

                        next_action = action

                next_q = target_net(torch.cat([state, next_action])).detach()

                target = reward + gamma * next_q.detach()

                loss = (best_q - target) ** 2

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

        if env.done:

            if env.winner == env.landlord:

                landlord_win += 1

                reward += 5

            else:

                reward -= 5

            target = reward

            loss = (best_q - target) ** 2

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    if (episode+1)%300==0:

        print('win=',landlord_win/300)

        print('loss:',loss,'reward:',reward)

        if landlord_win> best:

            best=landlord_win

            torch.save(net.state_dict(),'model/DQN_bellman.pth')

        winning_rate = landlord_win / 300

        landlord_win=0

        with open("graph/bellman_training_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, winning_rate])

df = pd.read_csv("graph/bellman_training_log.csv", header=None, names=["episode", "winning_rate"])

plt.plot(df["episode"], df["winning_rate"])
plt.xlabel("Episode")
plt.ylabel("Winning Rate")
plt.title("Bellman Training Win Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig("graph/bellman_training_curve_108state.png")
plt.show()

print('END')






















