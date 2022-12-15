# Playing Shengji with deep reinforcement learning

## Introduction
In recent years, humans have made significant progress in building AIs for perfect and imperfect information games. However, trick-taking poker games, which belong to the family of imperfect information games, are largely unsolved due to their complexity. Shengji (a.k.a. Tractor) is a 4-player trick-taking card game played with 2 decks of cards that involves competition, collaboration, and state and action spaces that are way larger than the vast majority of existing card games. Currently, to the best of my knowledge, there is no existing AI system that can play Shengji. In this work, I present `Shengji+`, the first AI system for the Shengji game powered by deep reinforcement learning and Monte Carlo methods. I open source my code to introduce Shengji as a new benchmark for imperfect information multi-agent RL, and to motivate future researchers on this topic.

## Installation
...

## Training
Running the system is as simple as
```shell
python TrainLoop.py --model-folder <SAVE_PATH>
```

There are many options that you can add to this training command. The current best result is produced by the command
```shell
python TrainLoop.py --model-folder --discount 0.95 --epsilon 0.01 --games 1500 --eval-size 300 --tutorial-prob 0.02 --dynamic-kitty --oracle-duration 50000
```

Documentation for the options will hopefully be added soon!

## Evaluation
You can evaluate the performance of a model by running
```shell
python TrainLoop.py --model-folder my/model/folder --eval-only --eval-size 3000
```

By default, the random agent is used as opponent. You can also specify the rule-based agent for evaluation by using the `--eval-agent strategic` option. If you would like to play with your model interactively, use the `--eval-agent interactive` option. In that case, you should also add the `--single-process` option so that the games are played in the main process.

To compare two models, you can run
```shell
python TrainLoop.py --eval-only --model-folder my/model1 --compare my/model2 --eval-size 1000
```

