[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/15965062/47237461-d2a90b00-d3e7-11e8-96a0-f0c9a0b7ad1d.png "Algorithm"
[image2]: add_plot_rewards "Plot of Rewards"

# Report - Deep RL Project: Continuous Control

### Implementation Details

The code for this project is ordered in 2 python files, 'ddpg_agent.py' and 'model.py', and the main training code and instructions in the notebook 'Continuous-Control.ipynb'. 

1. 'model.py': Architecture and logic for the neural networks implementing the actor and critic for the chosen DDPG algorithm.

2. 'ddpg_agent.py': Implements the agent class, which includes the logic for the stepping, acting, learning and the buffer to hold the experience data on which to train the agent, and uses 'model.py' to generate the local and target networks for the actor and critic.

3. 'Continuous-Control.ipynb': Main training logic and usage instructions. Includes explainations about the environment, state and action space, goals and final results. The main training loop creates an agent and trains it using the DDPG (details below) until satisfactory results. 

### Learning Algorithm

The agent is trained using the DDPG algorithm.

References:
1. [DDPG Paper](https://arxiv.org/pdf/1509.02971.pdf)

2. [DDPG-pendulum implementation](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)

3. Algorithm details: 

![Algorithm][image1]


4. Short explanation (refer to the papers for further details):
    - Q-Learning is not straighforwardly applied to continuous tasks due to the argmax operation over infinite actions in the continuous domain. DDPG can be viewed as an extension of Q-learning to continuous tasks.

    - DDPG was introduced as an actor-critic algorithm, although the roles of the actor and critic here are a bit different then the classic actor-critic algorithms. Here, the actor implements a current policy to deterministically map states to a specific "best" action. The critic implemets the Q function, and is trained using the same paradigm as in Q-learning, with the next action in the Bellman equation given from the actor's output. The actor is trained by the gradient from maximizing the estimated Q-value from the critic, when the actor's best predicted action is used as input to the critic.
    
    - As in Deep Q-learning, DDPG also implements a replay buffer to gather experiences from the agent (or the multiple parallel agents in the 2nd version of the stated environment). 
    
    - In order to encourage exploration during training, Ornstein-Uhlenbeck noise is added to the actors selected actions. I also needed to decay this noise using an epsilon hyperparameter to achieve best results.
    
    - Another fine detail is the use of soft updates (parameterized by tau below) to the target networks instead of hard updates as in the original DQN paper. 
    
6. Hyperparameters:

Parameter | Value
--- | ---
replay buffer size | int(1e6)
minibatch size | 256
discount factor | 0.99  
tau (soft update) | 1e-3
learning rate actor | 1e-3
learning rate critic | 1e-3
L2 weight decay | 0
UPDATE_EVERY | 20
NUM_UPDATES | 10
EPSILON | 1.0
EPSILON_DECAY | 1e-5

6. Network architecture:
    - Both the actor and critic are implemented using fully connected networks, with 2 hidden layers of 128 units each, batch normalization and Relu activation function, with Tanh activation at the last layer.
    - Input and output layers sizes are determined by the state and action space.
    - Training time until solving the environment takes around 90 minutes on AWS p2 instance with Tesla k80 GPU.
    - See 'model.py' for more details.

### Plot of results

As seen below, the agent solves the environment after ~135 episodes, and achieves best average score of above 37.

Episodes | Average Score | Max | Min | Time
--- | --- | --- | --- | ---
... | ... | ... | ... | ...
Episode 151 | Average Score: 33.59 | Max: 39.64 | Min: 30.34 | Time: 23.88
Episode 152 | Average Score: 33.76 | Max: 39.45 | Min: 30.88 | Time: 23.98
Episode 153 | Average Score: 33.92 | Max: 38.23 | Min: 20.35 | Time: 23.96
Episode 154 | Average Score: 34.08 | Max: 38.29 | Min: 27.17 | Time: 23.83
Episode 155 | Average Score: 34.24 | Max: 39.54 | Min: 30.72 | Time: 23.93
Episode 156 | Average Score: 34.40 | Max: 39.38 | Min: 34.30 | Time: 23.97
Episode 157 | Average Score: 34.55 | Max: 39.53 | Min: 34.66 | Time: 23.94
Episode 158 | Average Score: 34.69 | Max: 39.40 | Min: 31.07 | Time: 24.11
Episode 159 | Average Score: 34.85 | Max: 39.59 | Min: 33.47 | Time: 24.00
Episode 160 | Average Score: 35.00 | Max: 39.47 | Min: 30.89 | Time: 24.00
Episode 160	| Average Score: 35.00 | | |  

Environment solved in 137 episodes!	Average Best Score: 35.00

![Plot of Rewards][image2]

###  Ideas for future work

1. This DDPG implementation was very dependent on hyperparameter settings and random seed. Solving the environment using PPO, TRPO or D4PG might allow a more robust solution to this task.

2. Solving the more challenging [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler) environment using edited versions of these same algorithms. 

I'll at least do PPO and attempts to solve the Crawler environment after submission of this project (due to Udacity project submission rules).
