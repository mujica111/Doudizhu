from collections import Counter


def card_type_score(action):
    if action == ["pass"]:
        return 999

    c = Counter(action)

    if len(action) == 1:
        return 10

    if len(action) == 2 and len(c)==1:
        return 5

    if len(action) == 3 and len(c)==1:
        return 3

    if len(action) == 4 and len(c)==1:
        return 1

    return 8


def rule_agent(env):

    player = env.current_player

    actions = env.fast_all_legal_actions(
        env.hands[player]
    )


    # 如果只有一个动作
    if len(actions)==1:
        return actions[0]


    hand_len=len(env.hands[player])


    candidates=[]


    for action in actions:

        if action==["pass"]:
            continue


        score=0


        score -= len(action)*2


        # 大牌惩罚
        score += max(action)*0.05


        # 保留炸弹
        c=Counter(action)

        if len(action)==4 and len(c)==1:
            score += 50


        # 三带不要乱拆
        if len(action)>=3:
            score += 5


        # 快结束疯狂走牌
        if hand_len<=5:
            score -= len(action)*5


        candidates.append((score,action))


    if len(candidates)==0:
        return ["pass"]


    return min(
        candidates,
        key=lambda x:x[0]
    )[1]