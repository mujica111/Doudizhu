import torch
import torch.nn as nn

from RL2.agent.Level1_agent import rule1_agent
from RL2.environment import *

net=QNetwork()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

landlord_win = 0

best=0

for episode in range(1000000):

    gamma = 1

    env=Doudizhu()

    env.reset()

    trajectory=[]

    while not env.done:

        reward = 0

        final_action =None

        best_q=float('-inf')

        hand=env.hands[env.current_player]

        if env.current_player==env.landlord:

            for act in env.fast_all_legal_actions(hand):

                state = torch.tensor(env._get_state(), dtype=torch.float32)

                action = torch.tensor(env.convert_actions(act), dtype=torch.float32)

                q=net(torch.cat([state,action])).item()

                if best_q<q:

                    best_q=q

                    final_action=act

                    a=action

                    s=state
            if random.random()<0.1 and episode <=5000:

                final_action=random.choice(env.fast_all_legal_actions(hand))

                a = torch.tensor(env.convert_actions(final_action), dtype=torch.float)

                s = torch.tensor(env._get_state(), dtype=torch.float)

            trajectory.append((s,a))

            env.step(final_action)

        else:
            env.step(rule1_agent(env))

    if env.done:

        if env.winner == env.landlord:

            landlord_win += 1

            reward += 5

        else:

            reward -= 5

    for training in reversed(trajectory):

        q = net(torch.cat([(training[0].detach()), torch.tensor(training[1]).detach()]))

        loss = (q - reward*gamma) ** 2

        gamma*=0.99

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    if episode%500==1:

        print(landlord_win/500)

        print('loss:',loss)

        if landlord_win>best:

            best=landlord_win

            torch.save(net.state_dict(),'model/monta_carlo_best.pth')

        landlord_win = 0


























