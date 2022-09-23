# Defines all potential actions that a player can take during a game.
from .CardSet import CardSet, MoveType
from .utils import LETTER_RANK, Declaration, TrumpSuite

class Action:
    "The base class of all actions. Should never be used directly."

    def __repr__(self) -> str:
        return "Action()"

class DeclareAction(Action):
    "Reveal one or more cards in the dominant rank (or a pair of identical jokers) to declare or override the trump suite."
    def __init__(self, declaration: Declaration) -> None:
        self.declaration = declaration
    def __repr__(self) -> str:
        return f"DeclareAction({self.declaration.suite}, lv={self.declaration.level})"

class DontDeclareAction(Action):
    "Choose not to reveal any dominant rank card during the draw phase of the game."

    def __repr__(self) -> str:
        return "DontDeclareAction()"

# Usually all possible actions are explicitly computed when a player is about to take an action. But there are (33 -> 8) possible ways for the dealer to select the kitty, so we decompose this stage into 8 actions. Each step the dealer choose one card to discard. He does this 8 times.
class PlaceKittyAction(Action):
    "Discards a card and places it in the kitty (dealer or chaodi player)."
    def __init__(self, card: str, count=1) -> None:
        self.card = card
        self.count = count
    def __repr__(self) -> str:
        if self.count == 1:
            return f"Discard({self.card})"
        else:
            return f"Discard({self.card}, count={self.count})"

class ChaodiAction(Action):
    "Changes trump suite and swap cards with the kitty."
    def __init__(self, declaration: Declaration) -> None:
        self.declaration = declaration
    def __repr__(self) -> str:
        return f"ChaodAction({self.declaration})"

class DontChaodiAction(Action):
    "Skip chaodi."
    def __repr__(self) -> str:
        return f"DontChaodiAction()"

class LeadAction(Action):
    "Start a new round by placing a single, pair, or tractor. For now, we disable combinations."
    def __init__(self, move: MoveType) -> None:
        self.move = move
    
    def __repr__(self) -> str:
        return f"LeadAction(move={self.move})"

class FollowAction(Action):
    "Follow the pattern of a previous player. Everyone except the first player in a trick will need to follow the pattern."
    def __init__(self, player: str, cardset: CardSet) -> None:
        super().__init__(player)
        self.cardset = cardset
