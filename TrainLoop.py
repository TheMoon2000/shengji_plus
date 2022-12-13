import importlib
import logging
import pickle
import random
import shutil
from torch.multiprocessing import Process, Lock, Queue
import torch, os
from Simulation import Simulation
from agents.Agent import InteractiveAgent, RandomAgent, StrategicAgent
from agents.RLAgents import *
import argparse
import tqdm
import numpy as np
from networks.StateAutoEncoder import StateAutoEncoder

ctx = torch.multiprocessing.get_context('spawn')
global_main_queue = ctx.Queue(maxsize=30)
global_chaodi_queue = ctx.Queue(maxsize=30)
global_declare_queue = ctx.Queue(maxsize=30)
global_kitty_queue = ctx.Queue(maxsize=30)
global_coach_queue = ctx.Queue(maxsize=30)
actor_processes = []

# Parallelized data sampling
def sampler(idx: int, main_agent, declare_agent, kitty_agent, chaodi_agent, discount, decay_factor, global_main_queue, global_chaodi_queue, global_declare_queue, global_kitty_queue, global_coach_queue, combos, epsilon=0.01, reuse_prob=0, warmup_games=0, tutorial_prob=0.0, oracle_duration=0, coach_model=None):
    train_sim = Simulation(
        main_agent=main_agent,
        declare_agent=declare_agent,
        kitty_agent=kitty_agent,
        chaodi_agent=chaodi_agent,
        enable_combos=combos,
        discount=discount,
        epsilon=0.2,
        warmup_games=warmup_games,
        tutorial_prob=tutorial_prob,
        oracle_duration=oracle_duration,
        coach_model=coach_model
    )
    while True:
        local_main, local_declare, local_kitty, local_chaodi, local_coach = [], [], [], [], []
        for i in range(10):
            with torch.no_grad():
                while train_sim.step()[0]: pass
                local_main.extend(train_sim.main_history[:])
                local_declare.extend(train_sim.declaration_history[:])
                local_chaodi.extend(train_sim.chaodi_history[:])
                local_kitty.extend(train_sim.kitty_history[:])
                local_coach.extend(train_sim.initial_hand_history[:])
            
            train_sim.reset(reuse_old_deck=random.random() < reuse_prob)
        train_sim.epsilon = max(epsilon, train_sim.epsilon / decay_factor)
        global_main_queue.put(local_main)
        global_declare_queue.put(local_declare)
        global_chaodi_queue.put(local_chaodi)
        global_kitty_queue.put(local_kitty)
        global_coach_queue.put(local_coach)

def evaluator(idx: int, main_agent, declare_agent, kitty_agent, chaodi_agent, eval_main, eval_declare, eval_kitty, eval_chaodi, combos, eval_size, eval_results_queue: Queue, verbosity=logging.WARNING, learn_from_eval=False):
    logging.getLogger().setLevel(verbosity)
    eval_sim = Simulation(
        main_agent=main_agent,
        declare_agent=declare_agent,
        kitty_agent=kitty_agent,
        chaodi_agent=chaodi_agent,
        enable_combos=combos,
        eval_main=eval_main,
        eval_declare=eval_declare,
        eval_kitty=eval_kitty,
        eval_chaodi=eval_chaodi,
        eval=True,
        learn_from_eval=learn_from_eval
    )
    for _ in range(eval_size):
        with torch.no_grad():
            while eval_sim.step()[0]: pass
        opponent_index = int(eval_sim.game_engine.dealer_position in ['N', 'S'])
        opponents_won = eval_sim.game_engine.opponent_points >= 80
        win_index = int(opponents_won) if opponent_index == 1 else (1 - opponents_won)
        eval_results_queue.put((
            win_index,
            opponent_index,
            eval_sim.game_engine.opponent_points,
            abs(eval_sim.game_engine.final_defender_reward),
            eval_sim.main_history[:],
            eval_sim.declaration_history[:],
            eval_sim.chaodi_history[:],
            eval_sim.kitty_history[:]
        ))
        eval_sim.reset()
    exit(0)


def train(games: int, model_folder: str, eval_only: bool, eval_size: int, compare: str = None, discount=0.99, decay_factor=1.2, combos=False, verbose=False, random_seed=1, single_process=False, epsilon=0.01, use_hash_exploration=False, hash_length=16, model_type='mc', tau=0.995, kitty_agent='fc', eval_agent_type='random', learn_from_eval=False, reuse_old_deck_prob=0.0, warmup_games=0, tutorial_prob=0.0, oracle_duration=0, use_coach=False):
    os.makedirs(model_folder, exist_ok=True)
    torch.manual_seed(0)
    random.seed(random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hash counts
    if use_hash_exploration:
        hash_autoencoder = StateAutoEncoder(state_dimension=1196, hash_length=hash_length).to(device)
        if os.path.exists(f'{model_folder}/state_ae.pt'):
            hash_autoencoder.load_state_dict(torch.load(f'{model_folder}/state_ae.pt', map_location=device), strict=False)
            print("Using loaded model for hash autoencoder")
            hash_autoencoder.enabled |= True
        hash_autoencoder.share_memory()
    else:
        hash_autoencoder = None
    
    try:
        train_models = importlib.import_module(f"{model_folder}.Models".replace('/', '.'))
    except:
        train_models = importlib.import_module("networks.Models")
    
    declare_model = train_models.DeclarationModel().to(device)
    if os.path.exists(f'{model_folder}/declare.pt'):
        declare_agent = declare_model.load_state_dict(torch.load(f'{model_folder}/declare.pt', map_location=device), strict=False)
        print("Using loaded model for declaration")
    declare_agent = DeclareAgent('declare', declare_model, tau=tau)
    
    if kitty_agent == 'fc':
        kitty_model = train_models.KittyModel().to(device)
        if os.path.exists(f'{model_folder}/kitty.pt'):
            kitty_model.load_state_dict(torch.load(f'{model_folder}/kitty.pt', map_location=device), strict=False)
            print("Using loaded model for kitty")
        kitty_agent = KittyAgent('kitty', kitty_model, tau=tau)
    elif kitty_agent == 'argmax':
        kitty_model = train_models.KittyArgmaxModel().to(device)
        if os.path.exists(f'{model_folder}/kitty_argmax.pt'):
            kitty_model.load_state_dict(torch.load(f'{model_folder}/kitty_argmax.pt', map_location=device), strict=False)
            print("Using loaded model for kitty argmax")
        kitty_agent = KittyArgmaxAgent('kitty_argmax', kitty_model, tau=tau)
    elif kitty_agent == 'rnn':
        kitty_model = train_models.KittyRNNModel().to(device)
        if os.path.exists(f'{model_folder}/kitty_rnn.pt'):
            kitty_model.load_state_dict(torch.load(f'{model_folder}/kitty_rnn.pt', map_location=device), strict=False)
            print("Using loaded model for kitty rnn")
        kitty_agent = KittyRNNAgent('kitty_rnn', kitty_model, tau=tau)
    else:
        kitty_model = train_models.KittyRNNModel(rnn_type='lstm').to(device)
        if os.path.exists(f'{model_folder}/kitty_lstm.pt'):
            kitty_model.load_state_dict(torch.load(f'{model_folder}/kitty_lstm.pt', map_location=device), strict=False)
            print("Using loaded model for kitty lstm")
        kitty_agent = KittyRNNAgent('kitty_lstm', kitty_model, tau=tau)
    
    chaodi_model = train_models.ChaodiModel().to(device)
    if os.path.exists(f'{model_folder}/chaodi.pt'):
        chaodi_model.load_state_dict(torch.load(f'{model_folder}/chaodi.pt', map_location=device), strict=False)
        print("Using loaded model for chaodi")
    chaodi_agent = ChaodiAgent('chaodi', chaodi_model, tau=tau)
    main_model = train_models.MainModel(use_oracle=oracle_duration > 0).to(device)
    if os.path.exists(f'{model_folder}/main.pt'):
        main_model.load_state_dict(torch.load(f'{model_folder}/main.pt', map_location=device), strict=False)
        print("Using loaded model for main game")
    
    if model_type == 'mc':
        main_agent = MainAgent('main', main_model, hash_model=hash_autoencoder, tau=tau)
    elif model_type == 'dqn':
        value_model = train_models.ValueModel().to(device)
        if os.path.exists(f'{model_folder}/value.pt'):
            value_model.load_state_dict(torch.load(f'{model_folder}/value.pt', map_location=device))
            print("Using loaded value model")
        value_model.share_memory()

        main_agent = QLearningMainAgent('main', main_model, value_model, discount=discount, hash_model=hash_autoencoder, tau=tau)
    
    if use_coach:
        coach_model = Coach().to(device)
        if os.path.exists(f'{model_folder}/coach.pt'):
            coach_model.load_state_dict(torch.load(f'{model_folder}/coach.pt', map_location=device), strict=False)
            print("Using loaded model for coach network")
    else:
        coach_model = None
    
    declare_model.share_memory()
    kitty_model.share_memory()
    chaodi_model.share_memory()
    main_model.share_memory()
    coach_model.share_memory()

    eval_main, eval_kitty, eval_chaodi, eval_declare = None, None, None, None
    if compare:
        try:
            eval_models = importlib.import_module(f'{compare}.Models'.replace('/', '.'))
            with open(f'{compare}/state.pkl', mode='rb') as f:
                eval_state = pickle.load(f)
        except:
            eval_models = importlib.import_module("networks.Models")
        try:
            eval_main_model = eval_models.MainModel(use_oracle=eval_state['oracle_duration'] > 0).to(device)
        except:
            eval_main_model = eval_models.MainModel().to(device)
        eval_main_model.load_state_dict(torch.load(f'{compare}/main.pt', map_location=device), strict=False)

        if os.path.exists(f'{compare}/value.pt'):
            eval_value_model = eval_models.ValueModel().to(device)
            eval_value_model.load_state_dict(torch.load(f'{compare}/value.pt', map_location=device))
            print(f"Using loaded value model in {compare} for comparison")
            eval_value_model.share_memory()
            eval_main = QLearningMainAgent('main', eval_main_model, eval_value_model, discount=discount)
        else:
            print(f"Using main model in {compare} for comparison")
            eval_main = MainAgent('main', eval_main_model)
        
        # TODO: support kitty argmax in compare mode
        eval_kitty_model = eval_models.KittyModel().to(device)
        eval_kitty_model.load_state_dict(torch.load(f'{compare}/kitty.pt', map_location=device), strict=False)
        print(f"Using kitty model in {compare} for comparison")
        eval_kitty = KittyAgent("kitty", eval_kitty_model)

        eval_declare_model = eval_models.DeclarationModel().to(device)
        eval_declare_model.load_state_dict(torch.load(f'{compare}/declare.pt', map_location=device), strict=False)
        print(f"Using declare model in {compare} for comparison")
        eval_declare = DeclareAgent("declare", eval_declare_model)

        eval_chaodi_model = eval_models.ChaodiModel().to(device)
        eval_chaodi_model.load_state_dict(torch.load(f'{compare}/chaodi.pt', map_location=device), strict=False)
        print(f"Using chaodi model in {compare} for comparison")
        eval_chaodi = ChaodiAgent('chaodi', eval_chaodi_model)

        eval_main_model.share_memory()
        eval_kitty_model.share_memory()
        eval_declare_model.share_memory()
        eval_chaodi_model.share_memory()

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARN)
    
    if not eval_only:
        with open(f'{model_folder}/command.txt', mode='w') as f:
            f.write(' '.join(sys.argv))
    
    stats = []
    iterations = 0

    try:
        with open(f'{model_folder}/state.pkl', mode='rb') as f:
            state = pickle.load(f)
            iterations = state['iterations']
            main_agent.optimizer.load_state_dict(state['main_optim_state'])
            kitty_agent.optimizer.load_state_dict(state['kitty_optim_state'])
            declare_agent.optimizer.load_state_dict(state['declare_optim_state'])
            chaodi_agent.optimizer.load_state_dict(state['chaodi_optim_state'])
            if state.get('hash_model_optim_state', None) and hash_autoencoder:
                hash_autoencoder.optimizer.load_state_dict(state['hash_model_optim_state'])
            if isinstance(main_agent, QLearningMainAgent):
                main_agent.value_optimizer.load_state_dict(state['value_optim_state'])
            if coach_model:
                coach_model.optimizer.load_state_dict(state['coach_optim_state'])
            print(f"Using checkpoint at iteration {iterations}")
        with open(f'{model_folder}/stats.pkl', mode='rb') as f:
            stats = pickle.load(f)
            print(f"Continuing with saved stats")
    except:
        print("Starting new training session")
        shutil.copyfile('networks/Models.py', f'{model_folder}/Models.py')
    
    if not eval_only:
        processes_count = 8
        for i in range(1 if single_process else processes_count):
            actor = ctx.Process(target=sampler, args=(i, main_agent, declare_agent, kitty_agent, chaodi_agent, discount, decay_factor ** (1 / games), global_main_queue, global_chaodi_queue, global_declare_queue, global_kitty_queue, global_coach_queue, combos, epsilon, reuse_old_deck_prob, warmup_games // processes_count, tutorial_prob, oracle_duration // processes_count, coach_model))
            actor.start()
            actor_processes.append(actor)
            
            print(f"Spawned process {i}")
        
        if warmup_games > 0:
            print(f"Starting with {warmup_games // processes_count * processes_count} warmup games...")
    
  
    while True:
        if not eval_only:
            print(f"Training iterations {iterations * games}-{(iterations+1) * games}...")
            coach_losses = []
            for _ in tqdm.tqdm(range(0, games, 10)):
                # print('wait', global_main_queue.qsize(), global_declare_queue.qsize(), global_kitty_queue.qsize(), global_chaodi_queue.qsize())
                declare_batch = global_declare_queue.get()
                kitty_batch = global_kitty_queue.get()
                main_batch = global_main_queue.get()
                chaodi_batch = global_chaodi_queue.get()
                declare_agent.learn_from_samples(declare_batch)
                kitty_agent.learn_from_samples(kitty_batch)
                chaodi_agent.learn_from_samples(chaodi_batch)
                main_agent.learn_from_samples(main_batch)
                if coach_model:
                    coach_batch = global_coach_queue.get()
                    coach_X = torch.zeros((len(coach_batch), 564), device=device)
                    coach_Y = torch.zeros(len(coach_batch), device=device)
                    for i, (ih, outcome) in enumerate(coach_batch):
                        coach_X[i] = ih.dynamic_tensor.to(device)
                        coach_Y[i] = outcome
                    pred = coach_model(coach_X)
                    coach_loss = coach_model.loss_fn(pred, coach_Y)
                    coach_model.optimizer.zero_grad()
                    coach_loss.backward()
                    coach_model.optimizer.step()
                    print(coach_loss.cpu().detach().item())
                    coach_losses.append(coach_loss.cpu().detach().item())
            for agent in (declare_agent, kitty_agent, chaodi_agent, main_agent):
                torch.save(agent.model.state_dict(), f'{model_folder}/{agent.name}.pt')
            if coach_model:
                torch.save(coach_model.state_dict(), f'{model_folder}/coach.pt')
            print('main loss:', np.mean(main_agent.train_loss_history), 'declare loss:', np.mean(declare_agent.train_loss_history), 'kitty loss:', np.mean(kitty_agent.train_loss_history), 'chaodi loss:', np.mean(chaodi_agent.train_loss_history))
            if coach_loss:
                print('coach loss:', np.mean(coach_loss))
            
            if use_hash_exploration:
                torch.save(hash_autoencoder.state_dict(), f'{model_folder}/state_ae.pt')
                print('hash loss:', np.mean(main_agent.hash_loss_history), 'total count:', hash_autoencoder.hash_total)
                main_agent.hash_loss_history.clear()
                if not hash_autoencoder.enabled:
                    hash_autoencoder.enabled |= True
                    print("Turning on hash exploration bonus starting in next iteration...")
            
            if isinstance(main_agent, QLearningMainAgent):
                torch.save(main_agent.value_network.state_dict(), f'{model_folder}/value.pt')
                print("value loss:", np.mean(main_agent.value_loss_history))
                main_agent.value_loss_history.clear()
            
            main_agent.train_loss_history.clear()
            declare_agent.train_loss_history.clear()
            kitty_agent.train_loss_history.clear()
            chaodi_agent.train_loss_history.clear()
        
        print("Evaluating model performance...")
        if eval_agent_type == 'random':
            eval_agent = RandomAgent('random')
        elif eval_agent_type == 'strategic':
            eval_agent = StrategicAgent('strategic')
        else:
            eval_agent = InteractiveAgent('interactive')
        if single_process:
            eval_sim = Simulation(
                main_agent=main_agent,
                declare_agent=declare_agent,
                kitty_agent=kitty_agent,
                chaodi_agent=chaodi_agent,
                enable_combos=combos,
                eval_chaodi=eval_agent,
                eval_kitty=eval_agent,
                eval_declare=eval_agent,
                eval_main=eval_agent,
                eval=True
            )
            for _ in tqdm.tqdm(range(eval_size)):
                with torch.no_grad():
                    while eval_sim.step()[0]: pass
                eval_sim.reset()
            win_counts = eval_sim.win_counts
            level_counts = eval_sim.level_counts
            opposition_points = eval_sim.opposition_points
        else:
            eval_count = 5 if not eval_only else 12
            eval_size = max(1, eval_size // eval_count * eval_count) # Must be multiple of eval_count
            win_counts = [0, 0] # Defenders, opponents
            level_counts = [0, 0]
            opposition_points = [[], []]
            eval_queue = ctx.Queue()
            eval_actors = []
            for i in range(min(eval_size, eval_count)):
                actor = ctx.Process(target=evaluator, args=(i, main_agent, declare_agent, kitty_agent, chaodi_agent, eval_main, eval_declare, eval_kitty, eval_chaodi, combos, max(1, eval_size // eval_count), eval_queue, logging.DEBUG if verbose else logging.WARNING, learn_from_eval))
                actor.start()
                eval_actors.append(actor)
        
            # Collect the games where the agent has lost
            failed_main_history, failed_declare_history, failed_chaodi_history, failed_kitty_history = [], [], [], []
            accumulated_failed_games = 0
            with tqdm.tqdm(total=eval_size) as progress_bar:
                for i in range(eval_size):
                    win_index, opponent_index, points, levels, main_history, declare_history, chaodi_history, kitty_history = eval_queue.get()
                    win_counts[win_index] += 1
                    level_counts[win_index] += levels
                    opposition_points[opponent_index].append(points)
                    if learn_from_eval and win_index == 1: # index 0 means agent won, 1 means baseline model won
                        failed_main_history.extend(main_history)
                        failed_declare_history.extend(declare_history)
                        failed_chaodi_history.extend(chaodi_history)
                        failed_kitty_history.extend(kitty_history)
                        accumulated_failed_games += 1
                        if accumulated_failed_games == 10:
                            try:
                                global_main_queue.put(failed_main_history, block=False)
                            except: pass
                            try:
                                global_declare_queue.put(failed_declare_history, block=False)
                            except: pass
                            try:
                                global_chaodi_queue.put(failed_chaodi_history, block=False)
                            except: pass
                            try:
                                global_kitty_queue.put(failed_kitty_history, block=False)
                            except: pass
                            failed_main_history, failed_declare_history, failed_chaodi_history, failed_kitty_history = [], [], [], []
                            accumulated_failed_games = 0
                    progress_bar.update(1)
                
        
            for a in eval_actors:
                a.join()

        print('Win counts:', win_counts, 'level counts:', level_counts)
        print("Average opposition points:", np.mean(opposition_points[0]), np.mean(opposition_points[1]))
        
        iterations += 1

        if not eval_only:
            stats.append({
                "iterations": iterations * games,
                "win_counts": win_counts[0] / sum(win_counts),
                "level_counts": level_counts[0] / sum(level_counts),
                "avg_points": [np.mean(opposition_points[0]), np.mean(opposition_points[1])]
            })
            with open(f'{model_folder}/stats.pkl', mode='w+b') as f:
                pickle.dump(stats, f)
            with open(f'{model_folder}/state.pkl', mode='w+b') as f:
                pickle.dump({
                    'iterations': iterations,
                    'main_optim_state': main_agent.optimizer.state_dict(),
                    'declare_optim_state': declare_agent.optimizer.state_dict(),
                    'kitty_optim_state': kitty_agent.optimizer.state_dict(),
                    'chaodi_optim_state': chaodi_agent.optimizer.state_dict(),
                    'hash_model_optim_state': hash_autoencoder.optimizer.state_dict() if hash_autoencoder else None,
                    'value_optim_state': main_agent.value_optimizer.state_dict() if isinstance(main_agent, QLearningMainAgent) else None,
                    'coach_optim_state': coach_model.optimizer.state_dict() if coach_model else None,
                    'oracle_duration': oracle_duration
                }, f)
        else:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train loop')
    parser.add_argument('--games', type=int, default=500)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-size', type=int, default=300)
    parser.add_argument('--model-folder', type=str, default='pretrained')
    parser.add_argument('--model-type', type=str, default='mc', choices=['mc', 'dqn'])
    parser.add_argument('--compare', type=str, default='')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--decay-factor', type=float, default=1.2)
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--enable-combos', action='store_true')
    parser.add_argument('--single-process', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--hash-exploration', action='store_true')
    parser.add_argument('--hash-length', type=int, default=16)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--kitty-agent', type=str, default='fc', choices=['fc', 'argmax', 'rnn', 'lstm'])
    parser.add_argument('--eval-agent', type=str, default='random', choices=['random', 'interactive', 'strategic'])
    parser.add_argument('--learn-from-eval', action='store_true')
    parser.add_argument('--reuse-old-deck-prob', type=float, default=0)
    parser.add_argument('--warmup-games', type=int, default=0)
    parser.add_argument('--tutorial-prob', type=float, default=0.0)
    parser.add_argument('--oracle-duration', type=int, default=0)
    parser.add_argument('--use-coach', action='store_true')
    args = parser.parse_args()
    train(args.games, args.model_folder, args.eval_only, args.eval_size, args.compare, args.discount, args.decay_factor, args.enable_combos, args.verbose, args.random_seed, args.single_process, args.epsilon, args.hash_exploration, args.hash_length, args.model_type, args.tau, args.kitty_agent, args.eval_agent, args.learn_from_eval, args.reuse_old_deck_prob, args.warmup_games, args.tutorial_prob, args.oracle_duration, args.use_coach)