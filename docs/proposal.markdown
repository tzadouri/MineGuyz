---
layout: page
title: Proposal
permalink: /proposal/
---



<br />

## Summary of the Project


#### The project is inspired by the Fall Guys game, ranging from rotating objects to disappearing platforms, there are a frustrating amount of obstacles on the way to ensure that getting to the finish line is as troublesome a task as possible. If the player gets hit by an obstacle and falls off the grid, then it will be taken back to the beginning of the race line. There will be other players in the game as well as racing. The agent will be rewarded according to the position it’s able to crosses the line among other agents. The predominant goal is to finish first. 

***Input:***
The RGB pixels of the screen 

***Output:***
Actions the agent needs to take, such as move direction, jump, or increase the speed. 





<br />

## AI/ML Algorithms 

#### Predominantly the algorithms we plan to implement are deep convolutional neural networks and Deep Q Learning algorithms. Possibly A* for navigation will be utilized. Also Malmos API's depth map functionality will be used for object avoidance and image processing.   







<br />

## Evaluation Plan
    
    
***Qualitative:***

#### For the reinforcement learning sector of the project, the evaluation metrics would be for the agent to receive a maximum reward if it finishes the cross line first. It will receive less reward if it finishes second relative to first place, it will be rewarded less if finishes third relative to the second place, and etc. If it finishes fifth it will get no reward and this is the established baseline. If finishes sixth and below it will start getting an incrementally negative reward. The race is among ten players. The purpose is to maximize the reward. Considering each race as an episode, the possibility of over ~1000 iterations to make an improvement, and the improvement metric is finishing one spot ahead from the most recent highest standing. For the Convolutional Neural Net, a pixel-wise cross-entropy loss function will be used for the image segmentation. 

<br />
    
***Quantitative:***

#### For the RL algorithm, the initial sanity check is to see the agent being able to eventually finish the race regardless of the position it takes. This is considering void all the stable and moving obstacles. The moonshot case would be to see if it’s able to finish among the top five contesters. From a CNN standpoint, it needs to detect incoming objects and moving objects around as well and be able to slow down or move faster to find the nearest hiding locations to avoid the obstacles. Also, it needs to be able to detect the correct pathways instead of hitting the walls. 




<br />

## Appointment with the Instructor

#### Friday, October 23rd, 4:30 pm
