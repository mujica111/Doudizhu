
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
DOUZERO_ROOT = ROOT / "DouZero"

if str(DOUZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DOUZERO_ROOT))

from douzero.env.game import InfoSet
from douzero.evaluation.deep_agent import DeepAgent


LOCAL_TO_DZ = {**{i: i + 2 for i in range(1, 13)},13: 17,14: 20,15: 30,}
DZ_TO_LOCAL = {v: k for k, v in LOCAL_TO_DZ.items()}

POSITIONS = ("landlord", "landlord_down", "landlord_up")
EXPECTED_INPUT_SIZE = {
    "landlord": 501,
    "landlord_down": 612,
    "landlord_up": 612,
}


def to_douzero_cards(cards):
    return sorted(LOCAL_TO_DZ[c] for c in cards)
def to_local_cards(cards):
    return sorted(DZ_TO_LOCAL[c] for c in cards)
def to_douzero_action(action):
    if action is None or action == ["pass"] or action == []:
        return []
    return to_douzero_cards(action)
def to_local_action(action):
    if not action:
        return ["pass"]
    return to_local_cards(action)
def to_douzero_position(player, landlord):
    if player == landlord:
        return "landlord"
    if player == (landlord + 1) % 3:
        return "landlord_down"
    return "landlord_up"
    
def build_play_history(trace, landlord):
    card_play_action_seq = []
    played_cards = {position: [] for position in POSITIONS}
    last_move_dict = {position: [] for position in POSITIONS}
    last_two_moves = []
    last_move = []
    bomb_num = 0

    for player, action in trace:
        position = to_douzero_position(player, landlord)
        dz_action = to_douzero_action(action)

        card_play_action_seq.append(dz_action)
        last_two_moves.append(dz_action)
        if len(last_two_moves) > 2:
            last_two_moves.pop(0)

        if dz_action:
            played_cards[position].extend(dz_action)
            last_move_dict[position] = dz_action
            last_move = dz_action

            if len(dz_action) == 4 and len(set(dz_action)) == 1:
                bomb_num += 1
            elif sorted(dz_action) == [20, 30]:
                bomb_num += 1

    return (card_play_action_seq,played_cards,last_move_dict,last_move,last_two_moves,bomb_num,)


def build_infoset(env):
    player = env.current_player
    landlord = env.landlord
    position = to_douzero_position(player, landlord)

    (card_play_action_seq,played_cards,last_move_dict,last_move,last_two_moves,bomb_num,) = build_play_history(env.trace, landlord)

    infoset = InfoSet(position)
    infoset.player_position = position
    infoset.player_hand_cards = to_douzero_cards(env.hands[player])
    infoset.legal_actions = [to_douzero_action(action)for action in env.fast_all_legal_actions(env.hands[player])]
    infoset.card_play_action_seq = card_play_action_seq
    infoset.played_cards = played_cards
    infoset.last_move_dict = last_move_dict
    infoset.last_move = last_move
    infoset.last_two_moves = last_two_moves
    infoset.num_cards_left_dict = {to_douzero_position(i, landlord): len(env.hands[i])for i in range(3)}
    infoset.bomb_num = bomb_num
    infoset.three_landlord_cards = to_douzero_cards(env.three_landlord_cards)


    infoset.other_hand_cards = [card for i in range(3)if i != player for card in to_douzero_cards(env.hands[i])]
    infoset.all_handcards = [to_douzero_cards(env.hands[i]) for i in range(3)]

    return infoset


def _checkpoint_input_size(path):
    state = torch.load(str(path), map_location="cpu", weights_only=False)

    return state["dense1.weight"].shape[1]


def load_douzero_agent(position, model_path):

    actual_size = _checkpoint_input_size(path)
    expected_size = EXPECTED_INPUT_SIZE[position]

    return DeepAgent(position, str(path))


def load_all_douzero_agents(model_dir):
    model_dir = Path(model_dir).expanduser().resolve()

    return {
        "landlord": load_douzero_agent("landlord", model_dir / "landlord.ckpt"),
        "landlord_down": load_douzero_agent("landlord_down", model_dir / "DOU_landlord_down.ckpt"),
        "landlord_up": load_douzero_agent("landlord_up", model_dir / "DOU_landlord_up.ckpt"),
    }


def douzero_action(env, agent):
    infoset = build_infoset(env)
    action = agent.act(infoset)
    local_action = to_local_action(action)
    return local_action


def douzero_step_action(env, agents):
    position = to_douzero_position(env.current_player, env.landlord )
    return douzero_action(env, agents[position])
