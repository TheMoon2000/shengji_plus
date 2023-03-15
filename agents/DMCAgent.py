from collections import deque
import logging
import pickle
import random
import sys
from typing import Deque, List, Tuple
import numpy as np
import os
from env.CardSet import CardSet, MoveType

from .Agent import SJAgent, StageModule

sys.path.append('.')
from env.Actions import Action, ChaodiAction, DeclareAction, DontChaodiAction, DontDeclareAction, FollowAction, LeadAction, AppendLeadAction, EndLeadAction, PlaceAllKittyAction, PlaceKittyAction
from env.utils import ORDERING_INDEX, Stage, softmax
from env.Observation import Observation
from networks.Models import *

# A generic class that describes a stage module for the DMC agent.
class DMCModule(StageModule):
    def __init__(self, batch_size: int, tau=0.1) -> None:
        self.batch_size = batch_size # preferred batch size
        self.tau = tau # soft weight update parameter
        self._model: nn.Module = None # don't set directly
        self._eval_model: nn.Module = None # don't set directly
        self.loss_fn = nn.MSELoss()
        self.train_loss_history: List[float] = []
        self.optimizer: torch.optim.Optimizer = None
    
    # Use this function to load a pretrained model
    def load_model(self, model: nn.Module):
        self._model = model
        self._model.share_memory()
        self._eval_model = pickle.loads(pickle.dumps(model)).to(next(model.parameters()).device)
        self._eval_model.eval().share_memory()
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99, eps=1e-5)

    # Helper function to prepare `Observation` and `Action` objects into tensors
    # Returns a tensor: (batched observation-action pairs, batched rewards)
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        raise NotImplementedError

    # Training function
    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]]):
        splits = int(len(samples) / self.batch_size)
        for subsamples in np.array_split(samples, max(1, splits), axis=0):
            *args, rewards = self.prepare_batch_inputs(subsamples)
            pred = self._model(*args)
            loss = self.loss_fn(pred, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            if torch.isnan(loss):
                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight.data)
                self._model.apply(init_weights)
                print(f"Model {self} encountered nan, reset weights to random.")
            else:
                nn.utils.clip_grad_norm_(self._model.parameters(), 80)
                self.optimizer.step()
                self.train_loss_history.append(loss.detach().item())

        for param, target_param in zip(self._model.parameters(), self._eval_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def act(self, obs: Observation, epsilon=None, training=True):
        def reward(a: Action) -> torch.Tensor:
            *state_and_action, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self._eval_model(*state_and_action)
        
        if epsilon and random.random() < epsilon:
            return random.choice(obs.actions)
            # return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        else:
            # If in verbose mode, log actions and their probabilities in test time
            if not training and logging.getLogger().level == logging.DEBUG:
                logging.debug("Probability of actions:")
                sorted_actions = sorted(obs.actions, key=reward, reverse=True)
                for i, action in enumerate(sorted_actions):
                    logging.debug(f"{i:2}. {action} (reward={round(reward(action).cpu().item(), 4)})")
                return sorted_actions[0]
            else:
                return max(obs.actions, key=reward)

class DeclareModule(DMCModule):
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
        device = next(self._model.parameters()).device
        return x_batch.to(device), gt_rewards.to(device)


class KittyModule(DMCModule):
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
                # TODO: add kitty to state
            ])
            state_batch[i] = state_tensor
            action_batch[i] = ac.get_dynamic_tensor(obs.dominant_suit, obs.dominant_rank)
            gt_rewards[i] = rw
        device = next(self._model.parameters()).device
        return state_batch.to(device), action_batch.to(device), gt_rewards.to(device)


class ChaodiModule(DMCModule):
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu' # next(self._model.parameters()).device
        return x_batch.to(device), gt_rewards.to(device)


class MainModule(DMCModule):
    def __init__(self, batch_size: int, use_oracle: bool, tau=0.1) -> None:
        super().__init__(batch_size, tau)

        self.use_oracle = use_oracle

    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float, Observation]], training=True):
        if self.use_oracle:
            x_batch = torch.zeros((len(samples), 1089 + 108)) # additionally provide other players' hands
        else:
            x_batch = torch.zeros((len(samples), 1089 - 2 * 108))
        history_batch = torch.zeros((len(samples), 15, 436)) # Store up to last 15 rounds of history
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw, *_) in enumerate(samples):
            assert isinstance(ac, LeadAction) or isinstance(ac, AppendLeadAction) or isinstance(ac, EndLeadAction) or isinstance(ac, FollowAction)
            historical_moves, current_moves = obs.historical_moves_dynamic_tensor # obs.historical_moves_tensor
            cardset = ac.cardset
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

            if self.use_oracle and training:
                state_tensor = torch.cat([obs.oracle_cardsets, state_tensor])
            elif self.use_oracle:
                state_tensor = torch.cat([torch.zeros(108 * 3), state_tensor])

            x_batch[i] = torch.cat([state_tensor, ac.dynamic_tensor(obs.dominant_suit, obs.dominant_rank)])
            history_batch[i] = historical_moves
            gt_rewards[i] = rw
        device = next(self._model.parameters()).device
        return x_batch.to(device), history_batch.to(device), gt_rewards.to(device)


class DMCAgent(SJAgent):
    def __init__(self, name: str, use_oracle: bool) -> None:
        super().__init__(name)

        self.declare_module: DeclareModule = DeclareModule(batch_size=64)
        self.kitty_module: KittyModule = KittyModule(batch_size=32)
        self.chaodi_module: ChaodiModule = ChaodiModule(batch_size=32)
        self.main_module: MainModule = MainModule(batch_size=64, use_oracle=use_oracle)
    
    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]], stage: Stage):
        if stage == Stage.declare_stage:
            self.declare_module.learn_from_samples(samples)
        elif stage == Stage.kitty_stage:
            self.kitty_module.learn_from_samples(samples)
        elif stage == Stage.chaodi_stage:
            self.chaodi_module.learn_from_samples(samples)
        elif stage == Stage.main_stage:
            self.main_module.learn_from_samples(samples)
        else:
            raise NotImplementedError()
    
    def optimizer_states(self):
        return {
            'declare_optim_state': self.declare_module.optimizer.state_dict(),
            'kitty_optim_state': self.kitty_module.optimizer.state_dict(),
            'chaodi_optim_state': self.chaodi_module.optimizer.state_dict(),
            'main_optim_state': self.main_module.optimizer.state_dict(),
        }

    def load_optimizer_states(self, state):
        self.main_module.optimizer.load_state_dict(state['main_optim_state'])
        self.kitty_module.optimizer.load_state_dict(state['kitty_optim_state'])
        self.declare_module.optimizer.load_state_dict(state['declare_optim_state'])
        self.chaodi_module.optimizer.load_state_dict(state['chaodi_optim_state'])

    def load_models_from_disk(self, train_models):
        # Load models for DMC
        loaded_models = True
        declare_model: nn.Module = train_models.DeclarationModel().cuda()
        if os.path.exists(f'{self.name}/declare.pt'):
            declare_model.load_state_dict(torch.load(f'{self.name}/declare.pt', map_location='cuda'), strict=False)
            print("Using loaded model for declaration")
        else:
            loaded_models = False
        self.declare_module.load_model(declare_model)

        kitty_model: nn.Module = train_models.KittyModel().cuda()
        if os.path.exists(f'{self.name}/kitty.pt'):
            kitty_model.load_state_dict(torch.load(f'{self.name}/kitty.pt', map_location='cuda'), strict=False)
            print("Using loaded model for kitty")
        else:
            loaded_models = False
        self.kitty_module.load_model(kitty_model)

        chaodi_model: nn.Module = train_models.ChaodiModel().cuda()
        if os.path.exists(f'{self.name}/chaodi.pt'):
            chaodi_model.load_state_dict(torch.load(f'{self.name}/chaodi.pt', map_location='cuda'), strict=False)
            print("Using loaded model for chaodi")
        else:
            loaded_models = False
        self.chaodi_module.load_model(chaodi_model)

        try:
            with open(f'{self.name}/state.pkl', mode='rb') as f:
                state = pickle.load(f)
                self.main_module.use_oracle = state['oracle_duration'] > 0
            with open(f'{self.name}/stats.pkl', mode='rb') as f:
                stats = pickle.load(f)
                iterations = stats[-1]['iterations']
                print(f"Using checkpoint at iteration {iterations}")
            # If resuming from checkpoint, subtract iterations from oracle duration
            oracle_duration = max(0, state['oracle_duration'] - iterations)
            print(f"Resuming with remaining oracle duration {oracle_duration}")
        except Exception as e:
            print(e)
            loaded_models = False
        main_model: nn.Module = train_models.MainModel(use_oracle=self.main_module.use_oracle).cuda()
        if os.path.exists(f'{self.name}/main.pt'):
            main_model.load_state_dict(torch.load(f'{self.name}/main.pt', map_location='cuda'), strict=False)
            print("Using loaded model for main game")
        else:
            loaded_models = False
        self.main_module.load_model(main_model)
        
        return loaded_models, stats[-1]['iterations'] if loaded_models else 0

    def save_models_to_disk(self):
        torch.save(self.declare_module._model.state_dict(), self.name + '/declare.pt')
        torch.save(self.kitty_module._model.state_dict(), self.name + '/kitty.pt')
        if self.chaodi_module._model is not None:
            torch.save(self.chaodi_module._model.state_dict(), self.name + '/chaodi.pt')
        torch.save(self.main_module._model.state_dict(), self.name + '/main.pt')
    
    def clear_loss_histories(self):
        self.declare_module.train_loss_history.clear()
        self.kitty_module.train_loss_history.clear()
        self.chaodi_module.train_loss_history.clear()
        self.main_module.train_loss_history.clear()