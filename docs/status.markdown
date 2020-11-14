---
layout: page
title: Status
permalink: /status/
---



### Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/SaT4Ns7_akk" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<br />
### Summary 

#### The predominant purpose of this project is for the agent to go through a three-dimensional playing field with different themes of obstacles in each quarter of the map as it goes in a straight path to get to the finish line. This structure of the map is inspired by the battle royale-style game of Fall Guys. In the first sector, there are sets of poles or stacked blocks where the agent needs to learn to avoid as it moves forward and then it gets to a bridge with a river surrounding it where it needs to learn to cross the bridge. For this prototype, the agent is only able to move straight and turn left and right, as the obstacles will become more complicated and not particularly static, the jump functionality will be extended to the action list. The map is surrounded by glass walls where the agent needs to avoid touching in order to avoid negative reinforcement as the glass walls represent out of the map. The finish line is indicated at the end by the Redstone blocks where as soon as the agent arrives and touches it, a maximum reward is given and the game ends. The first prototype is a single agent game whereas in the future updates it is possibly going to be multi-agent. Currently, the benchmark for the single-agent is to avoid touching obstacles in order to maximize the reward when it arrives at the finish line. 

![My image Name](/assets/images/mineFallz.png)


<br />
### Approach 

#### For this prototype, we used the Deep Q-Learning Algorithm. Initially, we used the replay memory for training the DQN. It essentially stores the agent's observed transitions in which it allows to reuse the data. Through random sampling, the transition that builds up a batch is decorrelated which will greatly improve the DQN training procedure. The purpose of the algorithm is to maximize the discounted cumulative reward. With Q-learning, by utilizing the given function:

<br />
&ensp;<img src="https://render.githubusercontent.com/render/math?math=Q^*: State \times Action \rightarrow \mathbb{R}">

<br /> 

#### And by taking the action in a given state, then a policy can be constructed as such to maximize the reward as shown below.

<br />
&ensp;<img src="https://render.githubusercontent.com/render/math?math=\pi^*(s) = \arg\!\max_a \ Q^*(s, a)">

<br /> 

#### Since the information on the world is extremely limited and there is no access to the Q* , by utilizing convolutional neural networks as a function approximator we can construct one and train it to be similar Q* to. The Bellman equation given below is used as such that every Q* function for a policy obeys this equation.  
<br />
&ensp;<img src="https://render.githubusercontent.com/render/math?math=Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))">

<br /> 


**The actions consists of the following:**

```math
1. Moving south (Forward)
2. Moving west (Left Horizontally)
3. Moving east (Right Horizontally) 
```
**The state space is the following:**

```math
[-10,10] x [1,40] = 840
```

<br />
**Loss function:**

#### As mentioned above the purpose of Deep Q-learning is to find the Q function in order to construct a policy to maximize the necessary reward. For every epoch, the Q function gets updated by minimizing the loss function below which is essentially the difference between the two sides of the equality of the Bellman equation specified above, also known as the temporal difference error. 

<br />
&ensp;<img src="https://render.githubusercontent.com/render/math?math=\delta = Q(s, a) - (r + \gamma \max_a Q(s', a))">

<br /> 

**Reward Functions**

<br />
&ensp;<img src="https://render.githubusercontent.com/render/math?math=R(s)=  250\hspace{0.4cm} \text{Agent reaches destination}">
&ensp;<img src="https://render.githubusercontent.com/render/math?math=R(s)=  -1\hspace{0.4cm} \text{Agent touches the glass or the walls around the map or the glass on ground representing a river}">
&ensp;<img src="https://render.githubusercontent.com/render/math?math=R(s)=  -1\hspace{0.4cm} \text{Agent touches diamond block}">
&ensp;<img src="https://render.githubusercontent.com/render/math?math=R(s)=  -1\hspace{0.4cm} \text{Agent touches emerald_blck or the pole obstacles in the first phase}">
&ensp;<img src="https://render.githubusercontent.com/render/math?math=R(s)=  -1\hspace{0.4cm} \text{Agent touches gold_block or the pole obstacles in the first phase}">
&ensp;<img src="https://render.githubusercontent.com/render/math?math=R(s)=  -1\hspace{0.4cm} \text{Agent touches the Pink wool or the ground for initial state}">
&ensp;<img src="https://render.githubusercontent.com/render/math?math=R(s)=    1\hspace{0.4cm} \text{Every time the agent passes a quarter of the distance on the map vertically toward the finish line}">


<br /> 


#### For the Q-network, the model will be a convolutional neural network that takes the observation tensor’s first index, and then it will output the action size. Essentially the network is attempting to predict the expected return of each specific action for the given input. In terms of deciding which action should be chosen, the epsilon greedy policy is implemented where partly the model chooses the action and sometimes the actions is chosen by random probability of starting with the hpyerparameter epsilon start and decaying toward epsilon end. Then epsilon decay hyperparameter is used to manage the rate. The following figure is a demonstration of the flow of the program where action chosen by random or epsilon greedy is an input to the Malmo environment where next is step is returned. The results are recorded in the replay memory and optimizatin is implemented on every iteration where random batches from replay memory are selected for the new policy training.  



![My image Name](/assets/images/fig_approach.png)


<br />
### Evaluation




<br />
### Remaining Goals 



<br />
### Challenges 




<br />
### Resources Used

#### The resources utilized so far have been using Malmo’s API to simulate the environment and Pytroch for creating tensors, neural networks, and optimizers. Certain functionalities from assignment that were useful were derived for this project. The following resources were used to develop implementations or to develop ideas.


1. [Policy Gradient Reinforcement Learning in Pytorch](https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf)
2. [Building a DQN in Pytorch](https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435)
3. [Pytorch Reinforcement Learning](https://github.com/bentrevett/pytorch-rl)
4. [How does the Bellman equation work in Deep RL](https://towardsdatascience.com/how-the-bellman-equation-works-in-deep-reinforcement-learning-5301fe41b25a)