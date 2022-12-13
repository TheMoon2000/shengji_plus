from collections import deque
import logging
import pickle
import random
import sys
from typing import Deque, List, Tuple
import torch
import numpy as np
from env.CardSet import CardSet, MoveType

from networks.StateAutoEncoder import StateAutoEncoder


from .Agent import SJAgent

sys.path.append('.')
from env.Actions import Action, ChaodiAction, DeclareAction, DontChaodiAction, DontDeclareAction, FollowAction, LeadAction, PlaceAllKittyAction, PlaceKittyAction
from env.utils import ORDERING_INDEX, Stage, softmax
from env.Observation import Observation
from networks.Models import *

class DeepAgent(SJAgent):
    def __init__(self, name: str, model: nn.Module, batch_size=64, hash_model: StateAutoEncoder = None, tau=0.1, use_oracle=False) -> None:
        super().__init__(name)

        self.model = model
        self.eval_model = pickle.loads(pickle.dumps(self.model)).to(next(model.parameters()).device)
        self.eval_model.eval().share_memory()
        self.batch_size = batch_size
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99, eps=1e-5)
        self.loss_fn = nn.MSELoss()
        self.train_loss_history: List[float] = []
        self.hash_loss_history: List[float] = []
        self.hash_model = hash_model
        self.tau = tau
        self.use_oracle = use_oracle
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        raise NotImplementedError
    
    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]]):
        splits = int(len(samples) / self.batch_size)
        for subsamples in np.array_split(samples, max(1, splits), axis=0):
            *args, rewards = self.prepare_batch_inputs(subsamples)
            pred = self.model(*args)
            loss = self.loss_fn(pred, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            if torch.isnan(loss):
                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight.data)
                self.model.apply(init_weights)
                # pred = self.model(*args)
                # loss = self.loss_fn(pred, rewards)
                # self.optimizer.zero_grad()
                # loss.backward()
                print(f"Model {self} encountered nan, reset weights to random.")
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), 80)
                self.optimizer.step()
                self.train_loss_history.append(loss.detach().item())

            if self.hash_model is not None:
                hash_loss = self.hash_model.update(args[0]) # args[0] is the state+action tensor representation
                self.hash_loss_history.append(hash_loss)
        for param, target_param in zip(self.model.parameters(), self.eval_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class DeclareAgent(DeepAgent):
    def __init__(self, name: str, model: DeclarationModel, batch_size=64, tau=0.995) -> None:
        super().__init__(name, model, batch_size, tau=tau)
        self.model: DeclarationModel

    def act(self, obs: Observation, explore=False, epsilon=None, training=True):
        def reward(a: Action):
            x_batch, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self.eval_model(x_batch)
        
        if explore:
            return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        elif epsilon and random.random() < epsilon:
            return random.choice(obs.actions)
            # return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        else:
            return max(obs.actions, key=reward)
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        x_batch = torch.zeros((len(samples), 179))
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw) in enumerate(samples):
            assert isinstance(ac, DeclareAction) or isinstance(ac, DontDeclareAction), "DeclareAgent can only handle declare actions"
            state_tensor = torch.cat([
                obs.dynamic_hand_tensor, # (108,)
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.perceived_trump_cardsets, # (36,)
            ])
            x_batch[i] = torch.cat([state_tensor, ac.tensor])
            gt_rewards[i] = rw
        device = next(self.model.parameters()).device
        return x_batch.to(device), gt_rewards.to(device)


class KittyAgent(DeepAgent):
    def __init__(self, name: str, model: KittyModel, batch_size=32, tau=0.995) -> None:
        super().__init__(name, model, batch_size, tau=tau)
        self.model: KittyModel
    
    def act(self, obs: Observation, explore=False, epsilon=None, training=True):
        def reward(a: Action):
            state_batch, action_batch, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self.eval_model(state_batch, action_batch)
        
        if explore:
            return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        elif epsilon and random.random() < epsilon:
            return random.choice(obs.actions)
            # return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        else:
            return max(obs.actions, key=reward)
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        state_batch = torch.zeros((len(samples), 172))
        action_batch = torch.zeros(len(samples), dtype=torch.int)
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw) in enumerate(samples):
            assert isinstance(ac, PlaceKittyAction), "KittyAgent can only handle place kitty actions"
            state_tensor = torch.cat([
                obs.dynamic_hand_tensor, # (108,)
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.perceived_trump_cardsets, # (36,)
            ])
            state_batch[i] = state_tensor
            action_batch[i] = ac.tensor
            gt_rewards[i] = rw
        device = next(self.model.parameters()).device
        return state_batch.to(device), action_batch.to(device), gt_rewards.to(device)


class KittyRNNAgent(DeepAgent):
    def __init__(self, name: str, model: KittyRNNModel, batch_size=32, tau=0.995) -> None:
        super().__init__(name, model, batch_size, tau=tau)
    
    def act(self, obs: Observation, explore=False, epsilon=None, training=True):
        def reward(a: Action):
            state, hand, card, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self.eval_model(state, hand, card)
        
        if explore:
            return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        elif epsilon and random.random() < epsilon:
            return random.choice(obs.actions)
        else:
            return max(obs.actions, key=reward)
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        state_batch = torch.zeros((len(samples), 64))
        hand_batch = torch.zeros(len(samples), 33, dtype=torch.long)
        card_batch = torch.zeros(len(samples), 1, dtype=torch.long)
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw) in enumerate(samples):
            assert isinstance(ac, PlaceKittyAction), "KittyAgent can only handle place kitty actions"
            state_tensor = torch.cat([
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.perceived_trump_cardsets, # (36,)
            ])
            state_batch[i] = state_tensor
            hand_batch[i][-obs.hand.size:] = torch.tensor([ORDERING_INDEX[card] for card in obs.hand.card_list()])
            card_batch[i] = ORDERING_INDEX[ac.card]
            gt_rewards[i] = rw
        device = next(self.model.parameters()).device
        return state_batch.to(device), hand_batch.to(device), card_batch.to(device), gt_rewards.to(device)

class KittyArgmaxAgent(DeepAgent):
    def __init__(self, name: str, model: KittyArgmaxModel, batch_size=16, tau=0.995) -> None:
        super().__init__(name, model, batch_size, tau=tau)
        self.model: KittyArgmaxModel
    
    def act(self, obs: Observation, explore=False, epsilon=None, training=True):
        x_batch, action_batch, _ = self.prepare_batch_inputs([(obs, None, 0)])        
        
        if epsilon and random.random() < epsilon:
            exploit = False
            scores = torch.rand(33)
        else:
            exploit = True
            scores = self.eval_model.forward(x_batch, action_batch).squeeze()
                
        # Find the cards corresponding to the 8 largest indices
        hand_cards = obs.hand.card_list()
        chosen_cards = CardSet()
        for chosen_index in scores.topk(8).indices:
            chosen_cards.add_card(hand_cards[chosen_index])
        
        action = PlaceAllKittyAction(chosen_cards, scores, explore=not exploit)
        if exploit: # Do not use random distributions for training
            action.dist = scores

        return action
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, PlaceAllKittyAction, float]]):
        state_batch = torch.zeros((len(samples), 172))
        action_batch = torch.zeros((len(samples), 33), dtype=torch.long)
        logits = torch.zeros((len(samples), 33))
        for i, (obs, ac, rw) in enumerate(samples):
            # TODO: get (33,) reward
            state_tensor = torch.cat([
                obs.hand.tensor, # (108,)
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.perceived_trump_cardsets, # (36,)
            ])
            state_batch[i] = state_tensor

            hand_cards = obs.hand.card_list()
            action_batch[i] = torch.tensor([ORDERING_INDEX[card] for card in hand_cards])

            if ac is None or ac.dist is None: continue

            # First, find the indices of the chosen cards among the 33 possible cards. 1 = chosen, 0 = not chosen.
            chosen_indices = torch.zeros(33, dtype=torch.bool)
            kitty_cards = ac.cards.card_list()
            for card_index in range(chosen_indices.shape[0]):
                try:
                    kitty_index = kitty_cards.index(hand_cards[card_index])
                    chosen_indices[card_index] = True
                    kitty_cards.pop(kitty_index)
                except:
                    pass
            
            # Reweigh the predicted possibilites.
            # If rw > 0, multiply the probs of unchosen cards by 1 / (rw + 1) and allocate their probability evenly among the 8 chosen cards.
            # Otherwise, multiply the probs of chosen cards by 1 / (-rw + 1) and allocate their probability evenly among the 25 unchosen cards.
            # new_dist = torch.zeros(33)
            # old_dist = ac.dist.cpu()
            # if rw > 0:
            #     new_dist[~chosen_indices] = old_dist[~chosen_indices] / (rw + 1)
            #     new_dist[chosen_indices] = old_dist[chosen_indices] + torch.sum(old_dist[~chosen_indices]) * rw / (rw + 1) / 8
            # else:
            #     new_dist[chosen_indices] = old_dist[chosen_indices] / (-rw + 1)
            #     new_dist[~chosen_indices] = old_dist[~chosen_indices] + torch.sum(old_dist[chosen_indices]) * -rw / (-rw + 1) / 25

            logits[i][chosen_indices] = rw
        device = next(self.model.parameters()).device
        return state_batch.to(device), action_batch.to(device), logits.to(device)
    
    def learn_from_samples(self, samples: List[Tuple[Observation, PlaceAllKittyAction, float]]):
        splits = int(len(samples) / self.batch_size)
        for subsamples in np.array_split(samples, max(1, splits), axis=0):
            state, action, rewards = self.prepare_batch_inputs(subsamples)
            pred_dist = self.model(state, action)
            loss = -(pred_dist * rewards).sum(-1).mean()
            self.optimizer.zero_grad()
            loss.backward()
            if torch.isnan(loss):
                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight.data)
                self.model.apply(init_weights)
                # pred = self.model(*args)
                # loss = self.loss_fn(pred, rewards)
                # self.optimizer.zero_grad()
                # loss.backward()
                print(f"Model {self} encountered nan, reset weights to random.")
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), 80)
                self.optimizer.step()
                self.train_loss_history.append(loss.detach().item())

        for param, target_param in zip(self.model.parameters(), self.eval_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class ChaodiAgent(DeepAgent):
    def __init__(self, name: str, model: ChaodiModel, batch_size=32, tau=0.995) -> None:
        super().__init__(name, model, batch_size, tau=tau)
        self.model: ChaodiModel

    def act(self, obs: Observation, explore=False, epsilon=None, training=True):
        def reward(a: Action):
            x_batch, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self.eval_model(x_batch)

        if explore:
            return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        elif epsilon and random.random() < epsilon:
            return random.choice(obs.actions)
            # return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        else:
            return max(obs.actions, key=reward)
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        x_batch = torch.zeros((len(samples), 178))
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw) in enumerate(samples):
            assert isinstance(ac, ChaodiAction) or isinstance(ac, DontChaodiAction), "ChaodiAgent can only handle chaodi decisions"
            state_tensor = torch.cat([
                obs.dynamic_hand_tensor, # (108,)
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.perceived_trump_cardsets, # (36,)
            ])
            x_batch[i] = torch.cat([state_tensor, ac.tensor])
            gt_rewards[i] = rw
        device = next(self.model.parameters()).device
        return x_batch.to(device), gt_rewards.to(device)
     
    
class MainAgent(DeepAgent):
    def __init__(self, name: str, model: MainModel, batch_size=64, hash_model: StateAutoEncoder = None, tau=0.1) -> None:
        super().__init__(name, model, batch_size, hash_model, tau, use_oracle=model.use_oracle)
        self.model: MainModel
    
    def act(self, obs: Observation, explore=False, epsilon=None, training=True):
        def reward(a: Action):
            x_batch, history_batch, _ = self.prepare_batch_inputs([(obs, a, 0, None)])
            if self.hash_model and training:
                exploration_bonus = self.hash_model.exploration_bonus(x_batch)
            else:
                exploration_bonus = 0
            return self.eval_model(x_batch, history_batch) + exploration_bonus

        if explore:
            return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        elif epsilon and random.random() < epsilon:
            return random.choice(obs.actions)
            # return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        else:
            return max(obs.actions, key=reward)
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float, Observation]]):
        if self.use_oracle:
            x_batch = torch.zeros((len(samples), 1089 + 108)) # additionally provide other players' hands
        else:
            x_batch = torch.zeros((len(samples), 1089 - 2 * 108))
        history_batch = torch.zeros((len(samples), 15, 436)) # Store up to last 15 rounds of history
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw, _) in enumerate(samples):
            historical_moves, current_moves = obs.historical_moves_dynamic_tensor # obs.historical_moves_tensor
            cardset = ac.move.cardset if isinstance(ac, LeadAction) else ac.cardset
            state_tensor = torch.cat([
                obs.dynamic_hand_tensor, # (108,),
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.chaodi_times_tensor, # (4,)
                obs.points_tensor, # (80,)
                obs.unplayed_cards_dynamic_tensor, # (108,)
                current_moves, # (328,)
                obs.kitty_dynamic_tensor, # obs.kitty_tensor, # (108,)
                # obs.current_dominating_player_index, # (3,)
                obs.dominates_all_tensor(cardset), # (1,)
            ])

            if self.use_oracle:
                state_tensor = torch.cat([obs.oracle_cardsets, state_tensor])

            x_batch[i] = torch.cat([state_tensor, ac.dynamic_tensor(obs.dominant_suit, obs.dominant_rank)])
            history_batch[i] = historical_moves
            gt_rewards[i] = rw
        device = next(self.model.parameters()).device
        return x_batch.to(device), history_batch.to(device), gt_rewards.to(device)

# Q learning
class QLearningMainAgent(DeepAgent):
    def __init__(self, name: str, model: MainModel, value_model: ValueModel, batch_size=50, discount=0.99, hash_model: StateAutoEncoder = None, tau=0.995) -> None:
        super().__init__(name, model, batch_size, tau=tau)
        
        self.hash_model = hash_model
        self.discount = discount
        self.value_network = value_model
        self.value_optimizer = torch.optim.RMSprop(self.value_network.parameters(), lr=0.0001, alpha=0.99, eps=1e-5)
        self.value_loss_history = []
    
    def act(self, obs: Observation, explore=False, epsilon=None, training=True):
        def reward(a: Action):
            x_batch, history_batch, *_ = self.prepare_batch_inputs([(obs, a, 0, None)])
            if self.hash_model and training:
                exploration_bonus = self.hash_model.exploration_bonus(x_batch)
            else:
                exploration_bonus = 0
            return self.eval_model(x_batch, history_batch) + exploration_bonus

        if explore:
            return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        elif epsilon and random.random() < epsilon:
            return random.choice(obs.actions)
        else:
            return max(obs.actions, key=reward)

    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float, Observation]]):
        splits = int(len(samples) / self.batch_size)
        for subsamples in np.array_split(samples, max(1, splits), axis=0):
            state_and_action, history, rewards, next_state, next_history, terminals = self.prepare_batch_inputs(subsamples)

            # Update Q network
            with torch.no_grad():
                next_values = self.value_network.forward(next_state, next_history)
            
            pred_values = self.model.forward(state_and_action, history)
            target_values = rewards + self.discount * (1 - terminals) * next_values
            q_loss = self.loss_fn(pred_values, target_values)

            self.optimizer.zero_grad()
            q_loss.backward()
            self.optimizer.step()

            # Update Value network
            pred_values = self.value_network(state_and_action[:, :-108], history)
            v_loss = self.loss_fn(pred_values, target_values)

            self.value_optimizer.zero_grad()
            v_loss.backward()
            self.optimizer.step()

            self.value_loss_history.append(v_loss.detach().item())
            self.train_loss_history.append(q_loss.detach().item())

            if self.hash_model is not None:
                hash_loss = self.hash_model.update(state_and_action)
                self.hash_loss_history.append(hash_loss)
        
        for param, target_param in zip(self.model.parameters(), self.eval_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float, Union[Observation, None]]]):
        x_batch = torch.zeros((len(samples), 1196 - 3 * 108))
        next_state_batch = torch.zeros((len(samples), 1088 - 3 * 108))
        history_batch = torch.zeros((len(samples), 20, 436)) # Store up to last 15 rounds of history
        next_history_batch = torch.zeros((len(samples), 20, 436))
        gt_rewards = torch.zeros((len(samples), 1))
        terminals = torch.zeros(len(samples), 1)
        for i, (obs, ac, rw, next_obs) in enumerate(samples):
            historical_moves, current_moves = obs.historical_moves_tensor
            state_tensor = torch.cat([
                # obs.perceived_cardsets, # (432,)
                obs.hand.tensor,
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.chaodi_times_tensor, # (4,)
                obs.points_tensor, # (80,)
                obs.unplayed_cards_tensor, # (108,)
                current_moves, # (328,)
                obs.kitty_tensor, # (108,)
            ])
            x_batch[i] = torch.cat([state_tensor, ac.tensor])
            history_batch[i] = historical_moves
            
            if next_obs is not None:
                next_historical_moves, next_current_moves = next_obs.historical_moves_tensor
                next_state_tensor = torch.cat([
                    # next_obs.perceived_cardsets, # (432,)
                    next_obs.hand.tensor, # (108,)
                    next_obs.dealer_position_tensor, # (4,)
                    next_obs.trump_tensor, # (20,)
                    next_obs.declarer_position_tensor, # (4,)
                    next_obs.chaodi_times_tensor, # (4,)
                    next_obs.points_tensor, # (80,)
                    next_obs.unplayed_cards_tensor, # (108,)
                    next_current_moves, # (328,)
                    next_obs.kitty_tensor
                ])
                next_state_batch[i] = next_state_tensor
                next_history_batch[i] = next_historical_moves
            else:
                terminals[i] = 1
            gt_rewards[i] = rw
        device = next(self.model.parameters()).device
        return x_batch.to(device), history_batch.to(device), gt_rewards.to(device), next_state_batch.to(device), next_history_batch.to(device), terminals.to(device)