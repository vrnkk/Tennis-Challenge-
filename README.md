# Tennis-Challenge

In this challenge you develope an agent for the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) challenge. 

![Tennis Image](https://github.com/vrnkk/Tennis-Challenge-/blob/master/tennis_gif.gif)

# Problem description 

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
*  This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

# Description for implementation 
Eather you use the udacity workspace to train the model. Or you can setup your environment descriped like [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). I always use the workspace within udacity they are awesome. 

You can download your own environment and start implementing. At Udacity you also can use a provided workspace within the nanodegree setup. Usually I choose for this possibility. 

First you get started with the Tennis.iypnb to train your own agent. 

The entire instructions are also provided [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet) 
