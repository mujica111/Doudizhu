
import os
import csv
import random

from torch.distributions import Categorical, kl_divergence

from RL2.environment import *
from RL2.agent.adaptor import *


# ==================== Config ====================

EPISODES = 300
GAMES_PER_BATCH = 500
EVAL_INTERVAL = 5
EVAL_GAMES = 100

GAMMA = 0.99
GAE_LAMBDA = 0.95

SHAPING_COEF = 0.2
CLIP_EPS = 0.2
ENTROPY_COEF = 0.001
BC_KL_COEF = 0.05

PPO_EPOCHS = 4
MINIBATCH_SIZE = 256
ACTOR_LR = 3e-5
CRITIC_LR = 1e-3

BC_PATH = "model/landlord_agent500000.pth"

LATEST_PATH = "model/ppo_v5_latest.pth"
CRITIC_PATH = "model/ppo_v5_critic.pth"
BEST_PATH = "model/ppo_v5_best.pth"
LOG_PATH = "graph/ppo_v5_log.csv"

DEVICE = torch.device("mps" )

os.makedirs("model", exist_ok=True)
os.makedirs("graph", exist_ok=True)

print("Device:", DEVICE)




def load_actor_checkpoint(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    return model


def action_scores(model, state, action_vectors):
    return torch.stack([
        model(torch.cat((state, action))).reshape(())
        for action in action_vectors
    ])



actor = Actor().to(DEVICE)
load_actor_checkpoint(actor, BC_PATH)

# Frozen BC reference policy.
bc_actor = Actor().to(DEVICE)
load_actor_checkpoint(bc_actor, BC_PATH)
bc_actor.eval()

for parameter in bc_actor.parameters():
    parameter.requires_grad_(False)


critic = Critic().to(DEVICE)

actor_optimizer = torch.optim.Adam(
    actor.parameters(), lr=ACTOR_LR
)

critic_optimizer = torch.optim.Adam(
    critic.parameters(), lr=CRITIC_LR
)

farmer_agents = {
    "landlord_down": load_douzero_agent(
        "landlord_down", "Douzero/landlord_down.ckpt"
    ),
    "landlord_up": load_douzero_agent(
        "landlord_up", "Douzero/landlord_up.ckpt"
    ),
}


# ==================== Evaluation ====================

def evaluate(model, games=EVAL_GAMES):
    was_training = model.training
    model.eval()

    landlord_wins = 0

    with torch.no_grad():
        for _ in range(games):
            env = Doudizhu()
            env.reset()

            while not env.done:
                player = env.current_player

                if player == env.landlord:
                    state = torch.tensor(env._get_state(), dtype=torch.float32,device=DEVICE)

                    actions, vectors = env.fast_all_legal_actions(env)
                    scores = action_scores( model, state, vectors)

                    chosen_action = actions[
                        scores.argmax().item()
                    ]
                else:
                    position = to_douzero_position(
                        player, env.landlord
                    )

                    chosen_action = douzero_action(
                        env, farmer_agents[position]
                    )

                env.step(chosen_action)

            if env.winner == env.landlord:
                landlord_wins += 1

    model.train(was_training)
    return landlord_wins / games


if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "games",
            "train_win_rate",
            "eval_win_rate",
            "best_eval_win_rate",
            "decisions"
        ])

best_eval_rate = -1.0

for batch in range(EPISODES):
    actor.train()
    critic.train()

    batch_trace = []
    landlord_wins = 0

    for game in range(GAMES_PER_BATCH):
        env = Doudizhu()
        env.reset()

        game_trace = []
        current_record = None

        while not env.done:
            player = env.current_player

            if player == env.landlord:
                state = torch.tensor( env._get_state(), dtype=torch.float32,device=DEVICE )


                if current_record is not None:
                    previous = game_trace[current_record]

                    if previous["next_value"] is None:
                        with torch.no_grad():
                            previous["next_value"] = ( critic(state).reshape(()).item())

                actions, action_vectors = env.fast_all_legal_actions( env)

                with torch.no_grad():
                    scores = action_scores(actor, state, action_vectors )

                    dist = Categorical(logits=scores)
                    action_idx = dist.sample()

                    old_log_prob = dist.log_prob(action_idx)
                    old_value = critic(state).reshape(())

                game_trace.append({
                    "state": state.detach().cpu(),
                    "action_vectors": action_vectors.detach().cpu(),
                    "action_idx": action_idx.item(),
                    "old_log_prob": old_log_prob.item(),
                    "old_value": old_value.item(),


                    "segment_reward": 0.0,
                    "duration": 0,
                    "next_value": None,

                    "advantage": 0.0,
                    "return": 0.0
                })

                current_record = len(game_trace) - 1
                chosen_action = actions[action_idx.item()]

            else:
                position = to_douzero_position(player, env.landlord)

                chosen_action = douzero_action(env, farmer_agents[position])

            phi_before = env.potential()

            env.step(chosen_action)

            phi_after = 0.0 if env.done else env.potential()

            reward = SHAPING_COEF * (GAMMA * phi_after - phi_before)

            if env.done:
                if env.winner == env.landlord:
                    reward += 1.0
                    landlord_wins += 1
                else:
                    reward -= 1.0


            if current_record is not None:
                record = game_trace[current_record]

                record["segment_reward"] += (
                    GAMMA ** record["duration"]
                ) * reward

                record["duration"] += 1

            if env.done and current_record is not None:
                game_trace[current_record]["next_value"] = 0.0



        next_advantage = 0.0

        for record in reversed(game_trace):
            duration = record["duration"]
            next_value = record["next_value"]

            if next_value is None:
                next_value = 0.0

            discount = GAMMA ** duration
            trace_discount = (GAMMA * GAE_LAMBDA) ** duration

            delta = (
                record["segment_reward"]
                + discount * next_value
                - record["old_value"]
            )

            advantage = delta + trace_discount * next_advantage

            record["advantage"] = advantage
            record["return"] = advantage + record["old_value"]

            next_advantage = advantage

        batch_trace.extend(game_trace)


    advantages = torch.tensor([r["advantage"] for r in batch_trace],dtype=torch.float32)

    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

    for i, record in enumerate(batch_trace):
        record["advantage"] = advantages[i].item()



    indices = list(range(len(batch_trace)))

    for _ in range(PPO_EPOCHS):
        random.shuffle(indices)

        for start in range(0, len(indices), MINIBATCH_SIZE):
            minibatch = [
                batch_trace[i]
                for i in indices[start:start + MINIBATCH_SIZE]
            ]

            actor_losses = []
            critic_losses = []
            entropies = []
            kl_losses = []

            for record in minibatch:
                state = record["state"].to(DEVICE)
                action_vectors = record["action_vectors"].to(DEVICE)

                scores = action_scores(actor, state, action_vectors)

                current_dist = Categorical(logits=scores)

                idx = torch.tensor( record["action_idx"],dtype=torch.long, device=DEVICE)

                new_log_prob = current_dist.log_prob(idx)

                old_log_prob = torch.tensor(record["old_log_prob"],dtype=torch.float32,device=DEVICE)

                ratio = torch.exp(new_log_prob - old_log_prob)

                advantage = torch.tensor(
                    record["advantage"],
                    dtype=torch.float32,
                    device=DEVICE
                )

                loss1 = ratio * advantage

                loss2 = torch.clamp(ratio,1 - CLIP_EPS,1 + CLIP_EPS
                ) * advantage

                actor_losses.append(
                    -torch.minimum(loss1, loss2)
                )

                entropies.append(current_dist.entropy())

                # Frozen BC reference distribution.
                with torch.no_grad():
                    bc_scores = action_scores(
                        bc_actor, state, action_vectors
                    )

                    bc_dist = Categorical(logits=bc_scores)

                kl_losses.append(kl_divergence(current_dist, bc_dist))

                value = critic(state).reshape(())

                target = torch.tensor(
                    record["return"],
                    dtype=torch.float32,
                    device=DEVICE
                )

                critic_losses.append(
                    (value - target).pow(2)
                )

            ppo_loss = torch.stack(actor_losses).mean()
            entropy = torch.stack(entropies).mean()
            bc_kl = torch.stack(kl_losses).mean()
            critic_loss = torch.stack(critic_losses).mean()

            actor_loss = (
                ppo_loss
                + BC_KL_COEF * bc_kl
                - ENTROPY_COEF * entropy
            )

            actor_optimizer.zero_grad()
            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), 0.5
            )

            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                critic.parameters(), 0.5
            )

            critic_optimizer.step()

    # -------- Save latest model --------

    train_rate = landlord_wins / GAMES_PER_BATCH
    games_done = (batch + 1) * GAMES_PER_BATCH

    torch.save(actor.state_dict(), LATEST_PATH)
    torch.save(critic.state_dict(), CRITIC_PATH)

    print(
        f"Games: {games_done}, "
        f"train win rate: {train_rate:.2%}, "
        f"decisions: {len(batch_trace)}"
    )

    # -------- Independent greedy evaluation --------

    eval_rate = None

    if (batch + 1) % EVAL_INTERVAL == 0:
        eval_rate = evaluate(actor, EVAL_GAMES)

        print(
            f"Evaluation: {eval_rate:.2%}, "
            f"best: {max(best_eval_rate, eval_rate):.2%}"
        )

        if eval_rate > best_eval_rate:
            best_eval_rate = eval_rate
            torch.save(actor.state_dict(), BEST_PATH)

            print("Saved new best model:", BEST_PATH)

    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            games_done,
            round(train_rate * 100, 2),
            round(eval_rate * 100, 2) if eval_rate is not None else "",
            round(best_eval_rate * 100, 2),
            len(batch_trace)
        ])

print("Training complete.")
print("Best evaluation:", f"{best_eval_rate:.2%}")
