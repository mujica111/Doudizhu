import random
from collections import Counter
from itertools import combinations
import torch.nn as nn
import torch
from functools import lru_cache

class Actor(nn.Module):
    def __init__(self):
        super(Actor,self).__init__()
        size=128

        self.net=nn.Sequential(
            nn.Linear(63,size),
            nn.ReLU(),
            nn.Linear(size,size),
            nn.ReLU(),
            nn.Linear(size,1)
        )
    def forward(self,x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        size = 128

        self.net = nn.Sequential(
            nn.Linear(48, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, 1)
        )

    def forward(self, x):
        return self.net(x)

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        size= 128

        self.net=nn.Sequential(
            nn.Linear(63,size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, 1,
        ))

    def forward(self, x):
        return self.net(x)

class Doudizhu:
    def __init__(self):
        self.players = 3
        self.tiles = []
        self.hands = [[] for _ in range(self.players)]
        self.current_player = 0
        self.done = False
        self.pass_count=2
        self.last_action = None
        self.res=[]
        self.winner = None
        self.trace=[]
        self.played_card=[0]*15
        self.landlord_hand=20
        self.player1_hand=17
        self.player2_hand=17
        self.player0_played=[0]*15
        self.player1_played=[0]*15
        self.player2_played=[0]*15

    def potential(self):
        landlord_cards = len(self.hands[self.landlord])
        farmer1_cards = len(self.hands[1])
        farmer2_cards = len(self.hands[2])
        return -landlord_cards / 20 + 0.5 * (farmer1_cards / 17 + farmer2_cards / 17)


    def reset(self):
        self.landlord_hand = 20
        self.player1_hand = 17
        self.player2_hand = 17
        self.played_card=[0]*15
        self.pass_count = 2
        self.last_action = None
        self.landlord = 0
        self.tiles = self._create_tiles()
        random.shuffle(self.tiles)
        self.hands = [[] for _ in range(self.players)]
        self.trace=[]

        for _ in range(17):
            for player in range(self.players):
                self.hands[player].append(self.tiles.pop())
        self.three_landlord_cards = []
        for _ in range(3):
            card = self.tiles.pop()
            self.hands[self.landlord].append(card)
            self.three_landlord_cards.append(card)
        self.hands[0]=sorted(self.hands[0])
        self.hands[1]=sorted(self.hands[1])
        self.hands[2]=sorted(self.hands[2])
        self.done = False
        self.current_player = self.landlord

    def card_type(self,card:list):
        if card is None:
            return False
        dict1={}
        count=len(card)
        one=0
        two=0
        three=0
        four=0
        consecutive=1
        value=0
        for num in card:
            dict1[num] = dict1.get(num, 0) + 1
        for num in range(0,16):
            if value==1 and dict1.get(num,0)==1:
                consecutive+=1
            else:
                consecutive=1
            value=dict1.get(num, 0)
            if value==1:
                one+=1
            if value==2:
                two+=1
            if value==3:
                three+=1
            if value==4:
                four+=1
            if consecutive==count and count>=5 and 13 not in card and 14 not in card and 15 not in card:
                return 'spring'
            if value==1 and count==1:
                return 'one'
            if value==2 and count==2:
                return 'two'
            if value==3 and count==3:
                return 'three'
            if value==3 and count==4:
                return 'three to one'
            if two==1 and three==1 and count==5:
                return 'three to two'
            if four==1 and count==4:
                return 'bomb'
            if four==1 and count==6:
                return 'four to two'
            if four==1 and count==8 and two==2:
                return 'four to pairs'
            if count==2 and 14 in card and 15 in card:
                return 'joker'
        if two>=3 and one==0 and three==0 and four==0 and 13 not in card:
            if max(card)-min(card)==two-1:
                return 'liandui'
        if two >=0 and one >=0 and three >=2 and four == 0:
            if two>0 and one>0:
                return False
            threee=0
            save=-1
            card1 = list(sorted(set(card)))
            for cards in card1:
                if dict1.get(cards, 0)==3 and cards!=13:
                    threee+=1
                    if cards==save+1 or save==-1:
                       save=cards
                    else:
                        return False
                if dict1.get(cards, 0) == 3 and cards == 13:
                    return False

            if one==0 and two==0:
                return 'aeroplane:none'
            if one==threee or two==1:
                return 'aeroplane:one'
            if two==threee:
                return 'aeroplane:two'
        return False

    def step(self, action):
        reward = 0
        self.hands[self.current_player] = sorted(self.hands[self.current_player])

        if self.done:
            raise ValueError("Game is already over.")

        player = self.current_player

        self.trace.append((player, action[:]))


        if action == ["pass"]:
            self.pass_count += 1

            if self.pass_count >= 2:
                self.last_action = None

        else:
            if self.legal_actions(action):
                for i in action:
                    self.hands[player].remove(i)
            else:
                raise ValueError("Invalid action.")

            self.last_action = action[:]
            self.pass_count = 0

        if len(self.hands[player]) == 0:
            self.done = True
            self.winner = player
            reward += 0
        if action != ["pass"]:

            if player == 0:

               self.landlord_hand -= len(action)

            if player == 1:

                self.player1_hand-= len(action)

            if player == 2:

                self.player2_hand -= len(action)

        self.current_player = (self.current_player + 1) % self.players

        for card in action:
            if card!= 'pass':
                self.played_card[card-1] += 1

                if self.current_player == 0:
                   self.player0_played[card-1] += 1

                if self.current_player == 1:
                   self.player1_played[card-1] += 1

                if self.current_player == 2:
                   self.player2_played[card-1] += 1


        return self._get_state(), reward, self.done

    def legal_actions(self,actions):
        if actions==['pass']:
            if self.pass_count >= 2:
                return False
            else:
                return True
        if self.card_type(actions) is False:
            return False
        if self.last_action is None:
            return True
        if self.beat(actions):
            return True
        if self.card_type(self.last_action) != self.card_type(actions):
            return False
        else:
            if self.card_type(self.last_action)=='aeroplane:none' or self.card_type(self.last_action) == 'aeroplane:one'or self.card_type(self.last_action) == 'aeroplane:two'or self.card_type(self.last_action) == 'three to one'or self.card_type(self.last_action) == 'three to two'or self.card_type(self.last_action) == 'four to two' or self.card_type(self.last_action) == 'four to pairs':
                count=Counter(self.last_action)
                most_count_for_last=count.most_common(1)[0][0]
                count=Counter(actions)
                most_count_for_now=count.most_common(1)[0][0]
                if most_count_for_now>most_count_for_last:
                    return True
                else:
                    return False
            else:
                if len(self.last_action)!=len(actions):
                    return False
                value1,value2 = 0,0
                for num in actions:
                    value1+=num
                for num in self.last_action:
                    value2+=num
                if value1<=value2:
                    return False
                else:
                    return True
    def beat(self,actions):
        if self.last_action is None and self.card_type(actions) is True:
            return True
        if self.card_type(actions)=='joker':
            return True
        elif self.card_type(actions)=='bomb':
            if self.card_type(self.last_action) != 'joker'and self.card_type(self.last_action)!='bomb':
                return True
            if self.card_type(self.last_action)=='bomb':
                if actions[0]>self.last_action[0]:
                    return True
        return False

    def convert_actions(self,actions):
        list1=[0]*15
        for i in actions:
            if i == 'pass':
                return list1
            else:
                list1[i-1]+=1
        return list1


    def _create_tiles(self):
        tiles = []

        for tile in range(1, 14):
            for _ in range(4):
                tiles.append(tile)
        tiles.append(15)
        tiles.append(14)
        return tiles

    def _get_state(self):
            list1 = [0] * 15
            list2 = [0] * 15
            for i in self.hands[self.current_player]:
                list1[i - 1] += 1
            if self.last_action is not None:
                for i in self.last_action:
                     if i != 'pass':
                         list2[i - 1] += 1

            return list1 + self.played_card + list2 + [self.landlord] +[self.player1_hand] + [self.player2_hand]
    def fast_all_legal_actions(self,hand):
        res=[]
        if self.legal_actions(['pass']):
            res.append(['pass'])
        my_turn=False
        if self.last_action is None:
            my_turn=True
        type1=self.card_type(self.last_action)
        count1=Counter(hand)
        for i in count1:
            if count1[i]==4 and self.beat([i,i,i,i]):
                res.append([i,i,i,i])
        if count1.get(14,0)==1 and count1.get(15,0)==1:
            res.append([14,15])
        if type1=='one' or my_turn:
            for i in count1:
                if self.legal_actions([i]):
                    res.append([i])
        if type1=='two' or my_turn:
            for i in count1:
                if self.legal_actions([i,i]) and count1[i]>1:
                    res.append([i,i])
        if type1=='three' or my_turn:
            for i in count1:
                if self.legal_actions([i,i,i]) and count1[i]>2:
                    res.append([i,i,i])
        if type1=='three to one'or my_turn:
            for i in count1:
                if count1[i] > 2 and self.legal_actions([i, i, i,14]):
                    for j in count1:
                        if i!=j:
                            res.append(sorted([i,i,i,j]))
        if type1=='three to two' or my_turn:
            for i in count1:
                if count1[i] > 2 and self.legal_actions([i, i, i,14,14]):
                    for j in count1:
                        if i!=j and count1[j]>1:
                            res.append(sorted([i,i,i,j,j]))
        if type1=='four to two' or my_turn:
            values = list(count1)
            for i in count1:
                if count1[i] >= 4 and self.legal_actions([i, i, i, i, 14, 14]):
                    for index, j in enumerate(values):
                        if j == i:
                            continue
                        for k in values[index:]:
                            if k == i:
                                continue
                            if j == k and count1[j] < 2:
                                continue
                            res.append(sorted([i, i, i, i, j, k]))
        if type1=='four to pairs'or my_turn:
            values = list(count1)
            pairs = []
            for i in values:
                if count1[i] >= 2:
                    pairs.append(i)
            for i in count1:
                if count1[i] >= 4 and self.legal_actions([i, i, i, i, 14, 14,15,15]):
                    for index,pair in enumerate(pairs):
                        if pair != i:
                            for second_pair in pairs[index+1:]:
                                if second_pair != i and second_pair != pair:
                                    ans=([i, i, i, i, pair, second_pair,pair,second_pair])
                                    res.append(sorted(ans))

        if type1=='spring'or my_turn:
            for i in range(0,13):
                path=[]
                copy=i
                while count1.get(copy, 0) > 0:
                    path.append(copy)
                    copy+=1
                    if len(path)>=5 and self.legal_actions(path):
                        res.append(path[:])
        if type1 == 'liandui' or my_turn:
            for i in range(0, 13):
                path = []
                copy = i
                while count1.get(copy, 0) > 1:
                    path.append(copy)
                    path.append(copy)
                    copy += 1
                    if len(path) >= 6 and self.legal_actions(path):
                        res.append(path[:])

        if type1=='aeroplane:none' or my_turn:
            for i in range(0, 13):
                path = []
                copy = i
                while count1.get(copy, 0) > 2:
                    path.extend([copy,copy,copy])
                    copy += 1
                    if len(path) >= 6 and self.legal_actions(path):
                        res.append(path[:])
        if type1 == 'aeroplane:one' or my_turn:
            bodies=[]
            wings=[]
            for i in range(0, 13):
                path = []
                copy = i
                while count1.get(copy, 0) > 2:
                    path.extend([copy, copy, copy])
                    copy += 1
                    if len(path) >= 6 :
                        bodies.append(path[:])
            for body in bodies:
                wings=hand[:]
                for card in body:
                    wings.remove(card)
                length=len(body)//3
                for wing in combinations(wings,length):
                    action = body + list(wing)
                    if self.legal_actions(action):
                         res.append(sorted(action))

        if type1 == 'aeroplane:two' or my_turn:
            bodies=[]
            wings=[]
            for i in range(0, 13):
                path = []
                copy = i
                while count1.get(copy, 0) > 2:
                    path.extend([copy, copy, copy])
                    copy += 1
                    if len(path) >= 6 :
                        bodies.append(path[:])
            for body in bodies:
                wings = hand[:]
                for card in body:
                    wings.remove(card)
                body_values = set(body)
                pairs = []
                for x in set(wings):
                    if x not in body_values and wings.count(x) >= 2:
                        pairs.append(x)
                length = len(body) // 3
                for wing in combinations(pairs, length):
                    action = body[:]
                    for x in wing:
                        action.extend([x, x])
                    if self.legal_actions(action):
                        res.append(sorted(action))
        res1=[]
        for res in res:
            if res not in res1:
                res1.append(res)


        return res1


    def min_play_count(self, hand):
            cards = [card for card in hand if card not in (1, 13, 14, 15)]
            if len(cards) < 2:
                return [0.0]
            mean = sum(cards) / len(cards)
            return [(sum((card - mean) ** 2 for card in cards) / len(cards)) ** 0.5]

    def hand_continuity(self, hand):
        ranks = sorted(set(card for card in hand if 1 <= card <= 11))
        if not ranks:
            return [0, 0]

        segments = 1
        longest = current = 1

        for i in range(1, len(ranks)):
            if ranks[i] == ranks[i - 1] + 1:
                current += 1
            else:
                segments += 1
                current = 1

            longest = max(longest, current)
        list1=[segments, longest]


        return list1

rule=Doudizhu()




