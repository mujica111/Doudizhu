import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from RL2.agent.adaptor import *
from RL2.environment import *
import csv

import pandas as pt
import matplotlib.pyplot as plt

landlord_agent= load_douzero_agent('landlord','Douzero/landlord.ckpt')

farmer_agents = {
    "landlord_down": load_douzero_agent("landlord_down", "opponents/down/sl_landlord_down.ckpt"),
    "landlord_up": load_douzero_agent("landlord_up", "opponents/up/sl_landlord_up.ckpt"),
}

env=Doudizhu()

trace=[]

trace1=[]

for game in range(1,200001):

    env.reset()

    while not env.done:

        player = env.current_player

        position = to_douzero_position(player, env.landlord)

        if env.current_player == 0:

            action=douzero_action(env,landlord_agent)


        else:

            action= douzero_action(env, farmer_agents[position])

            if env.current_player == 1:

                state = env._get_state()

                trace1.append([action, state, env.fast_all_legal_actions(env.hands[player])])



            if env.current_player == 2:

                state = env._get_state()

                trace.append([action, state, env.fast_all_legal_actions(env.hands[player])])


        env.step(action)

    if game%50000==0 and game!=0:

        torch.save(trace, f'BC_training/sl_farmer2_agent{game}.pth')

        torch.save(trace1, f'BC_training/sl_farmer1_agent{game}.pth')

        print(f'saved:{game}')







