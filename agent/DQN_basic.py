import torch
import torch.nn as nn

from RL2.agent.Level1_agent import rule1_agent
from RL2.environment import *

net=QNetwork()

optimizer=torch.optim.Adam(net.parameters())

landlord_win=0

best=0

for episode in range(10000):

    env=Doudizhu()

    env.reset()

    while not env.done:

        reward = 0

        final_action =None

        best_q=float('-inf')

        hand=env.hands[env.current_player]

        if env.current_player==env.landlord:

            for act in env.fast_all_legal_actions(hand):

                action= torch.tensor(env.convert_actions(act),dtype=torch.float)

                state= torch.tensor(env._get_state(),dtype=torch.float)

                q=torch.cat([state,action])

                q_value=net(q)

                if best_q<q_value:

                    best_q=q_value

                    final_action=act

            if random.random()<0.1 :

                final_action=random.choice(env.fast_all_legal_actions(hand))

                action = torch.tensor(env.convert_actions(final_action), dtype=torch.float)

                best_q = net(torch.cat([state, action]))

            reward+=len(final_action)*0.05

            if final_action==['pass']:

                reward-=0.02

            env.step(final_action)

            if env.done:

                if env.winner==env.landlord:

                    landlord_win += 1

                    reward += 5

                else:

                    reward -= 5

            loss = (best_q - reward) ** 2

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        else:
            env.step(rule1_agent(env))

    if episode%300==0:

        print('win=',landlord_win/300)

        print('loss:',loss,'reward:',reward)

        if landlord_win>best:

            best=landlord_win

            torch.save(net.state_dict(),'model/DQN_basic.pth')

        landlord_win=0



print(landlord_win)



















