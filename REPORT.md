[//]: # (Image References)

[image1]: add_image "Algorithm"
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

    - DDPG was introduced as an actor-critic algorithm, although the roles of the actor and critic here are a bit different then the classic actor-critic algorithms. The actor implements a current policy to deterministically map states to a specific "best" action. The critic implemets the Q function, and is trained using the same paradigm as in Q-learning, with the next action in the Bellman equation given from the actor's output.
    
    - As in Deep Q-learning, DDPG also implements a replay buffer to gather experiences from the agent (or the multiple parallel agents in the 2nd version of the stated environment). In order to encourage exploration during training, Ornstein-Uhlenbeck noise is added to the actors selected action.  
    
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

6. Network architecture:
    - Both the actor and critic are implemented using fully connected networks, with 2 hidden layers of 128 units each, batch normalization and Relu activation function, with Tanh activation at the last layer.
    - Input and output layers sizes are determined by the state and action space.
    - Training time until solving the environment (~135 episodes) takes around 90 minutes on AWS p2 instance with Tesla k80 GPU.
    - See 'model.py' for more details.

### Plot of results

As seen below, the agent solves the environment after ~135 episodes, and achieves best average score of above 37.

![Plot of Rewards][image2]

###  Ideas for future work

1. This DDPG implementation was very dependent on hyperparameter settings and random seed. Solving the environment using PPO or TRPO might allow a more robust solution to this task.

2. Solving the more challenging [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler) environment using edited versions of these same algorithms. 
