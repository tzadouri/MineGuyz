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

#### Qualitative

<h4>Since the goal of the agent is to make it to the end of the obstacle course with minimal collisions, and in the least number of steps, we can qualitatively see the agent learning as the episodes progress. In the first few episodes, the agent gets stuck a few times behind some obstacles, and is not able to make it to the redstone finish line within the specified max step limit. However after about 10 episodes it starts to make it to the end and achieves the highest reward of making it to the end. After which the agent seeks to minimize the steps, since each step is a slight penalty, to maximize overall reward and find the path that is most like a straight path to the finish line.</h4>



#### Quantitative

<h4>
  Below we can see the reward return graph chart. The goal of the agent is to make it to the end with minimal collisions with any of the obstacles. For each section in the obstacle course the agent is rewarded based on the amount of difficulty it is to reach past each threshold. The agent gets a penalty each time it hits an obstacle or goes off path. Furthermore, the agent is incentivized to reach the redstone finish line, due to the finish line having the highest reward of any threshold. We have also made progression forward a lot faster in terms of steps for each in the action dictionary, so the agent would progress forward more than anything else. The reward chart above details how the agent learns as the episodes progress. We can see that after a while in training the agent's return goes down, this can perhaps be due to the neural network overtraining and learning specific patterns and not being general enough for different randomly general obstacle courses.
</h4>

![graph1](assets/images/graph1.png)

<h4>
Later we adjusted the reward system for the agent. We removed any reward for moving forwards, instead changing to there being a penalty for being for each step taken. We then set the rewards for each subsection to be 100, and 150, bringing them a lot closer together. We then set the penalty for going into the water to -10 and for hitting any obstacle to -1. After implementing these changes we then restrained our model and were pleased with the results. We began to see the rewards increase constantly and not decrease and training went on beyond a certain threshold.
Below is the reward graph for the newly trained agent.
</h4>

![graph1](assets/images/graph2.png)

<h4>
For this training attempt we also graphed the episode loss and we can see that it consistently began to converge towards 0 and the steps went towards positive infinity.
</h4>

![graph1](assets/images/Loss.png)

<br />

### Remaining Goals 

<h4>As of this status report we have yet to add the 3rd stage for our agent to pass through. For this stage we are thinking of something a bit more complex, perhaps a mage of bridges for the agent to learn its way through as well as maybe some enemies, we are currently drafting ideas and want to make this stage special. We would also like to implement some type of computer vision as well, most likely using some type of Pytorch library to do so, we believe this will significantly improve the accuracy and training speed of our agent as well as make its movements far more realistic. Per the recommendation of the professor we are also looking at adding some type of a depth map to our project as well, apparently with the help of a built in Malmo library. This will make the agent far more aware of just what is surrounding it and its overall distance from the finish line. Furthermore we hope to put this added complexity of the agent to good use in our more intricate third phase, which we are currently drafting ideas for.</h4>

 <h4>Our final goal for this project is to ultimately be able to have multiple agents compete for who can get to the end the fastest. This will take full advantage of the upgrades that we are making to our agent in the coming weeks. Each agent will be given a fixed amount of steps and time to be trained in and its parameters with respect to a deep convolutional neural network will be exported to a text file. During the competition each agent will be configured to its trained state and will then be set free against the other agents. This competition will be displayed in our final video and report.</h4>

<br />

### Challenges 

<h4>A challenge that we are currently facing is deciding upon what exactly we want our final stage to be like, we are trying to implement some type of dynamic environment that moves at a constant speed, however we are facing challenges with regards to implementing this in Malmo as well as what approach we might need in order to train the agent in order to deal with a dynamically moving environment.
</h4>

<h4>
Some of the other challenges we might face will be implementing our desired features in the amount of time that we have left. We had hoped to get computer vision working by the time this report was due, however due to the amount of time it took to get used to working with Malmo beyond the the basic homework, i.e creating the environment, and training and DQNN to allow the agent to make it to the very end, we ran out of time, although we have looked deeply into it. Now knowing the fundamentals of Malmo we hope to be able to be more productive with time hence why our goals for the second half are more ambitious. If we get stuck we might ask our favorite TA Kolby for any advice or implementation strategies.
</h4>

<h4>
Lastly another challenge that we might face in the multi agent implementation. We had originally desired to have 10 or so agents compete at once for who could make it to the end in a myriad of different levels. However we may run into limitations with regards to training time and hardware resource limits as to how many agents we can have and how many different fields.
</h4>



<br />

### Resources Used

#### The resources utilized so far have been using Malmo’s API to simulate the environment and Pytroch for creating tensors, neural networks, and optimizers. Certain functionalities from assignment that were useful were derived for this project. The following resources were used to develop implementations or to develop ideas.


1. [Policy Gradient Reinforcement Learning in Pytorch](https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf)
2. [Building a DQN in Pytorch](https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435)
3. [Pytorch Reinforcement Learning](https://github.com/bentrevett/pytorch-rl)
4. [How does the Bellman equation work in Deep RL](https://towardsdatascience.com/how-the-bellman-equation-works-in-deep-reinforcement-learning-5301fe41b25a)