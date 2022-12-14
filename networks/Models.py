from turtle import forward
from typing import Union
import torch
import torch.nn as nn
from torch.optim import RMSprop
import sys

class DeclarationModel(nn.Module):
    "The declaration model's observation includes: player's cards, player's position relative to the dealer, current declaration (rank and suite), position of current declaration, known trump cards in each player's hands."
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(172 + 7, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass of declaration reward prediction. The input tensor should have shape (B, 179), where B is the batch dimension. The first 172 values are for the observation, and last 7 are for the action.
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

class KittyModel(nn.Module):
    "The kitty model's observation includes: the player's cards, the player's position relative to the dealer, current declaration, position of current declaration, known trump cards in each player's hand (same as those in DeclarationModel)."
    def __init__(self, dynamic_kitty=False) -> None:
        super().__init__()

        self.dynamic_kitty = dynamic_kitty
        self.single_card_embedding = nn.Embedding(54, 54)
        self.fc1 = nn.Linear(172 + 54, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor, card: torch.Tensor):
        """
        Performs the forward pass of kitty placement reward prediction. The input tensors should have shape (B, 172) and (B,), where B is the batch dimension. The first 172 values are for the observation, and the last B values are the indexes of the card to discard.
        """

        if self.dynamic_kitty:
            card_embeddings = torch.zeros((x.shape[0], 54), device=x.device)
            card_embeddings[torch.arange(x.shape[0], dtype=torch.long), card.long()] = 1
        else:
            card_embeddings = self.single_card_embedding(card)

        x = self.fc1(torch.hstack([x, card_embeddings]))
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

class KittyRNNModel(nn.Module):
    "The kitty model's observation includes: the player's cards, the player's position relative to the dealer, current declaration, position of current declaration, known trump cards in each player's hand (same as those in DeclarationModel)."
    def __init__(self, rnn_type='lstm') -> None:
        super().__init__()

        self.rnn_type = rnn_type
        self.card_embedding = nn.Embedding(54, 64)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.in1 = nn.Linear(64, 72)
        self.in2 = nn.Linear(72, 128)


        self.out1 = nn.Linear(128, 128)
        self.out2 = nn.Linear(128, 1)
    
    def forward(self, x: torch.Tensor, hand: torch.Tensor, card: torch.Tensor):
        """
        Performs the forward pass of kitty placement reward prediction. The input tensors should have shape (B, 172) and (B,), where B is the batch dimension. The first 172 values are for the observation, and the last B values are the indexes of the card to discard.
        """

        sequence =  self.card_embedding(torch.cat([hand, card], dim=-1))
        initial_state = self.in1(x)
        initial_state = torch.relu(initial_state)
        initial_state = self.in2(initial_state)

        if self.rnn_type == 'gru':
            _, h_n = self.rnn.forward(sequence, initial_state.unsqueeze(0))
        else:
            _, (h_n, _) = self.rnn.forward(sequence, (initial_state.unsqueeze(0), initial_state.unsqueeze(0)))
        x = self.out1(h_n.squeeze(0))
        x = torch.relu(x)
        x = self.out2(x)
        return x

class KittyArgmaxModel(nn.Module):
    "The kitty model's observation includes: the player's cards, the player's position relative to the dealer, current declaration, position of current declaration, known trump cards in each player's hand (same as those in DeclarationModel)."
    def __init__(self) -> None:
        super().__init__()

        self.single_card_embedding = nn.Embedding(54, 10)
        self.fc1 = nn.Linear(172 + 33 * 10, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 33)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor, cards: torch.Tensor):
        """
        Performs the forward pass of kitty placement reward prediction. The input tensors should have shape (B, 172) and (B,), where B is the batch dimension. The first 172 values are for the observation, and the last B values are the indexes of the card to discard.
        """

        card_embeddings = self.single_card_embedding(cards).view((x.shape[0], -1))

        x = self.fc1(torch.hstack([x, card_embeddings]))
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return self.softmax(x)

class ChaodiModel(nn.Module):
    "The chaodi model's observation includes: the player's cards, the player's position relative to the dealer, current declaration, position of current declaration, known trump cards in each player's hand."
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(178, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass of chaodi decision reward prediction. The input tensor should have shape (B, 178), where B is the batch dimension. The first 172 values are for the observation, and the last 6 values are the action.
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

class MainModel(nn.Module):
    """
    The main model's observation includes:
        - The player's perceived cardsets for all players based on incomplete information (432)
        - The player's position relative to the dealer
        - The current declaration
        - The position of the current declaration
        - The number of times each player chaodied
        - Points earned by opponents and points escaped by defenders
        - Unplayed cards of shape (108)
        - Historical moves of shape (H, 436)
        - Cards played in the current round by every other player, and the position of the player that is leading the round
        - Kitty (if the player is the last to place the kitty), shape (108,)
    """
    def __init__(self, use_oracle=False) -> None:
        super().__init__()

        self.use_oracle = use_oracle
        self.lstm = nn.LSTM(436, 256, batch_first=True)
        if use_oracle:
            self.fc1 = nn.Linear(1197 + 256, 768)
        else:
            self.fc1 = nn.Linear(1197 - 3 * 108 + 256, 768)
        self.fc2 = nn.Linear(768, 512)
        self.fc_rest = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.fc_final = nn.Linear(512, 1)
    
    def forward(self, x: torch.Tensor, history: torch.Tensor):
        # history_out, _ = self.lstm(history)
        # history_out = history_out[:, -1, :]
        _, (history_out, _) = self.lstm(history) # experimental \
        
        x = self.fc1(torch.hstack([x, history_out.squeeze(0)]))
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc_rest(x)
        x = self.fc_final(x)
        return x

class ValueModel(nn.Module):
    """
    The value model's observation includes:
        - The player's perceived cardsets for all players based on incomplete information (432)
        - The player's position relative to the dealer
        - The current declaration
        - The position of the current declaration
        - The number of times each player chaodied
        - Points earned by opponents and points escaped by defenders
        - Unplayed cards of shape (108)
        - Historical moves of shape (H, 436)
        - Cards played in the current round by every other player, and the position of the player that is leading the round
        - Kitty (if the player is the last to place the kitty), shape (108,)
    """
    def __init__(self) -> None:
        super().__init__()

        self.lstm = nn.LSTM(436, 256, batch_first=True)
        self.fc1 = nn.Linear(1088 - 3 * 108 + 256, 768)
        self.fc2 = nn.Linear(768, 512)
        self.fc3 = nn.Linear(512, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        # self.fc4 = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor, history: torch.Tensor):
        if history.shape[1] > 0:
            history_out, _ = self.lstm(history)
            history_out = history_out[:, -1, :]
        else:
            history_out = torch.zeros((x.shape[0], 256))
        x = self.fc1(torch.hstack([x, history_out]))
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        # x = torch.relu(x)
        # x = self.fc4(x)
        return x

class Coach(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.loss_fn = nn.BCELoss()

        self.fc = nn.Sequential(
            nn.Linear(564, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.optimizer = RMSprop(self.parameters(), lr=0.0001, alpha=0.99, eps=1e-5)
    
    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        return x.squeeze(-1)