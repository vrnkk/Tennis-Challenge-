# Report for the Tennis Challenge in the Udacity Depp Reninforcement Learning Nanodegree

## Project Description 
At [Unity ML-Agents toolkit](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md) you can find the [Tennis Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).
In this environment two tennis rackets keep the ball in the game. They bounce it over the net. Udacity adapted the environemnt a bit and did a challenge for their students. There are diffrences to the original but still you can adapt a lot of thoughts and things. 

## Basic Information
![Tennis Image](https://github.com/vrnkk/Tennis-Challenge-/blob/master/tennis_gif.gif)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.
* The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Algorithm: Deep Deterministic Policy Gradient method
Because of the continuous action space, classical value-based methods are hard to apply. Here the environment is solved using the Multiagent Deep Deterministic Policy Gradient (DDPG) method. This is a flavour of an actor-critic method, where the actor is trained to maximize the expected outcome as is predicted by the critic network.

Here critics also see the observation space and next actions of the opponents. During traing this is allowed. When it´s about testint the agents the actors only relies on the local observation. Then you don´t need the critics anymore.

What me helped to understand it besides the great Udacity explanation are this sources: 
 * [First](http://proceedings.mlr.press/v32/silver14.pdf)
 * [Second](https://www.youtube.com/watch?v=_pbd6TCjmaw)
 * [Third](https://arxiv.org/pdf/1706.02275.pdf)
 * [Fourth](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)

## Implementation
The approach here was to turn the algorithm given [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) into a multi-agent algorithm. There have therefore been some changes. 

The original code only has a DDPG agent with actor and critic. In our environment, we have two agents each with their own actor and critic and own observations.

Computing the next actions is pretty straightforward, as the actor only needs to know about its own current state.

More complicated is the improvement of the critic. This also relies on the opponent ´s observations and actions. Great publication from [OpenAI](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)

This has affects to the Q targets. You also need to incorporate the actor_target of the other actor. That also applies when you want to compute the current actor s loss. THen you also need the critic that takes the other actors action into account.

### Noise 
Noise is handled a bit diffently. It´s scaled down by a factor and multiplied in every episode. So the results ia a gradually decreasing noise level and shift from exploration to exploitation by more training runs. 
By testing there is no noise. 

## Deep Neural Network architecture
For both actor and critic, we use linear neural networks with two hidden layers with Relu activation functions. Their sizes are 265 and 128 nodes.

The input layer of the actor has 24 nodes, as this corresponds to the state size. The input layer of the critic has 52 nodes, as this corresponds to 2 * (state size + action size).

For the actor, the output layer has 2 nodes, which corresponds to the action size, and a tanh-activation function to map to the interval (-1, 1). The critic has naturally only one output node without activation function to allow for any numerical output value.

You need a bach normalization layer between the hidden layers. Otherwise the agents will not solve the environment. Possibly because of the noise. 

## Hyperparameters
* the discount factor gamma=0.99
* the batch size is 512
* the actor learning rate is 1e-3 and the critic is 1e-2
* every 4 time steps we train both agents with one batch from the replay buffer.
* the noise is scaled with a factor of 0.7 and gradually reduce the noise by a factor of 0.9999 

## Performance 
The training took quite a while. So I stopped with testing hyperparameter combinations. Here you can see the performance results for the hyperparemeters mentioned aboce. 

![Performance of the approach](.png)
Performance of the score over the episodes. 
It was solved in .... 

## Ideas to Improve it
  * Hyperparameter Tuning
  * Regularize the NN to get a smoother behaviour of the agents

There are for sure plenty of other things to improve the network. 
