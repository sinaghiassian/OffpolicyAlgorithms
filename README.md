<p align="center">
    <img width="100" src="/Assets/rlai.png" />
</p>

<h2 align=center>Off-policy Prediction Learning Algorithms</h2>
<div align="center">
  :steam_locomotive::train::train::train::train::train:
</div>
This repository ... 


<p align="center">
    <img src="/Assets/fourRoomGridWorld.gif" />
    <img src="/Assets/chain.gif" />
</p>

## Table of Contents
- **[Algorithms](#algorithms)**: [Off-policy TD](#td), [GTD](#gtd), [Emphatic TD](#ETD), [Emphatic TDβ](#ETDB)
- **[Environments](#environment)** :  [Chain](#chain), [Four Room Grid World](#four_room_grid_world)
- **[Tasks](#tasks)** : [Collision](#collision), [Hallway proximity](#hallway_proximity), 
  [High variance hallway proximity](#highvar_hallway_proximity)
- **[Glossary]**(#glossary)

## Run
- [Learning.py](#learning.py)
- [Job Buidler](#job_builder)




<a name='algorithms'></a>
## Algorithms
Algorithms are used to find a weight vector, **w**, such that the dot product of **w** and the feature vector, 
approximates the value function. 

<a name='td'></a>

### Off-policy TD

**Paper** [Off-Policy Temporal-Difference Learning with Function Approximation](https://www.cs.mcgill.ca/~dprecup/publications/PSD-01.pdf)<br>
**Author** Doina Precup, Richard S. Sutton, Sanjoy Dasgupta<br>

#### Main update rule:
```python
def learn_wights(s, s_p, r):
        delta = compute_delta(s, s_p, r, gamma)
        w += alpha * delta * z
```
where s and s_p are the current and next states, r is the reward, and gamma is the discount factor parameter



<a name='environment'></a>
## Environment
At the heart of an environment is an MDP.
The MDP defines the states, actions, rewards, transition probability matrix, and the discount factor.

<a name="four_room_grid_world"></a>

### Four Room Grid World

<a name="four_room_grid_world"></a>

### Chain

<a name='tasks'></a>
## Tasks
A task, or a problem, uses an environment along with a target and behavior policy.
With this definition, multiple tasks could be defined on one environment.

## How to Run the Code

The code can be run in two different ways.
One way is through *learning.py* that can be used to run small experiments on a local computer.
The other way is through the files inside the Job directory. 
We explain each of these approaches below.

<a name="learning.py"></a>
### Learning.py
```sh
$ learning.py -p1 p1
```

<a name="job_builder"></a>
### Job Builder


