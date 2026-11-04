import torch
from RL2.environment import *
import random

env=Doudizhu()

Actor=Actor()

optimizer = torch.optim.Adam(Actor.parameters(), lr=1e-4)

epochs=100

batch_size=500

data = torch.load("BC_training/landlord_agent100000.pth", map_location="cpu", weights_only=False)

BC_trace = []

for action, state, legal_actions in data:

    for i, actions in enumerate(legal_actions):

        if actions == action:

            BC_trace.append([i, state, legal_actions])

expert_choice = []

idx=0

count=0

BC_training_set=[]

for training_set in BC_trace:

    idx,state,legal_actions = training_set[0],training_set[1],training_set[2]

    state=torch.tensor(state,dtype=torch.float)

    action_set = []

    for act in legal_actions:

        action=torch.tensor(env.convert_actions(act), dtype=torch.float)

        action_set.append(action)

    BC_training_set.append((state,action_set,idx))

loss_fn = nn.CrossEntropyLoss()

for epoch_idx in range(epochs):

    print("Epoch {}".format(epoch_idx))

    random.shuffle(BC_training_set)

    epoch_loss = 0.0

    batch_count = 0

    for start in range(0, len(BC_training_set), batch_size):

        batch = BC_training_set[start:start + batch_size]

        B = len(batch)

        max_actions = max(len(actions) for state, actions, idx in batch)

        features = torch.zeros(B, max_actions, 63)

        lengths = []

        targets = []

        for row, (state, actions, idx) in enumerate(batch):

            lengths.append(len(actions))

            targets.append(idx)

            for col, action in enumerate(actions):

                features[row, col] = torch.cat([state, action])

        logits = Actor(features.reshape(-1, 63)).reshape(B, max_actions)

        lengths = torch.tensor(lengths)

        mask = torch.arange(max_actions).unsqueeze(0) >= lengths.unsqueeze(1)

        logits = logits.masked_fill(mask, float("-inf"))

        targets = torch.tensor(targets, dtype=torch.long)

        loss = loss_fn(logits, targets)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        batch_count += 1

torch.save(Actor.state_dict(), 'model/landlord_agent100000_100_epoch.pth')









































