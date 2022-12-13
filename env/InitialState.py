from typing import List
from env.CardSet import CardSet
from env.utils import AbsolutePosition, Declaration, TrumpSuite
import torch

class InitialState:
    def __init__(self, dealer_hand: CardSet, opposite_hand: CardSet, left_hand: CardSet, right_hand: CardSet, dominant_rank: int, declaration: Declaration, kitty: CardSet, chaodi_times: List[int], dealer_position: AbsolutePosition) -> None:
        self.dealer_hand = dealer_hand
        self.opposite_hand = opposite_hand
        self.left_hand = left_hand
        self.right_hand = right_hand
        self.dominant_rank = dominant_rank
        self.declaration = declaration
        self.kitty = kitty
        self.chaodi_times = chaodi_times
        self.dealer_position = dealer_position
    
    @property
    def dominant_suit(self):
        return self.declaration.suite if self.declaration else TrumpSuite.XJ
    
    @property
    def trump_tensor(self):
        "Returns a (20,) tensor representing the current trump suit and trump rank."
        rank_tensor = torch.zeros(13)
        rank_tensor[self.dominant_rank - 2] = 1

        return torch.cat([self.declaration.tensor if self.declaration else torch.zeros(7), rank_tensor])
    
    @property
    def chaodi_times_tensor(self):
        dealer_idx = ['N', 'W', 'S', 'E'].index(self.dealer_position)
        return torch.tensor(self.chaodi_times[dealer_idx:] + self.chaodi_times[:dealer_idx])

    @property
    def dynamic_tensor(self):
        return torch.cat([
            self.dealer_hand.get_dynamic_tensor(self.dominant_suit, self.dominant_rank),
            self.opposite_hand.get_dynamic_tensor(self.dominant_suit, self.dominant_rank),
            self.left_hand.get_dynamic_tensor(self.dominant_suit, self.dominant_rank),
            self.right_hand.get_dynamic_tensor(self.dominant_suit, self.dominant_rank),
            self.trump_tensor,
            self.kitty.get_dynamic_tensor(self.dominant_suit, self.dominant_rank),
            self.chaodi_times_tensor
        ])
