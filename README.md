<p align="center">
    <img width="100" src="/Assets/rlai.png" />
</p>

<h2 align=center>Off-policy Prediction Learning Algorithms</h2>
<div align="center">
  :steam_locomotive::train::train::train::train::train:
</div>
This repository includes the code for the "empirical off-policy" paper.


<p align="center">
    <img src="/Assets/fourRoomGridWorld.gif" />
    <img src="/Assets/chain.gif" />
</p>

## Table of Contents
- **[How to run the code](#how-to-run)**: [Learning.py](#learning.py), [Job Buidler](#job_builder)
- **[Algorithms](#algorithms)**
    - **[Algorithm Glossary](#glossary)**
    - TD: [Off-policy TD](#td)
    - Gradient-TD family : [GTD](#gtd) , [GTD](#gtd2), [HTD](#htd), [PGTD2](#pgdt2), [TDRC](#tdrc)
    - Emphatic-TD family: [Emphatic TD](#etd), [Emphatic TDβ](#etdb)  
    - Variable-λ family: [TB](#tb), [Vtrace](#vtrace), [ABTD](#abtd)
    - Least squared family: [LSTD](#lstd), [LSETD](#lsetd)
- **[Environments](#environment)** :  [Chain](#chain), [Four Room Grid World](#four_room_grid_world)
- **[Tasks](#tasks)** : [Collision](#collision), [Hallway proximity](#hallway_proximity), 
  [High variance hallway proximity](#highvar_hallway_proximity)



<a name='how-to-run'></a>
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




<a name='algorithms'></a>
## Algorithms
Algorithms are used to find a weight vector, **w**, such that the dot product of **w** and the feature vector, 
approximates the value function. 

<a name='glossary'></a>
### Algorithm Glossary
Here, we briefly explain all the symbols and variables names that we use in our implementation.

#### meta-parameters
- Common parameters of all algorithms:
  - Alpha (α): is the step size that defines how much the weight vector `w` is updated at each time step.
  - Lambda (λ): is the bootstrapping parameter.
- Common parameters of Gradient-TD algorithms:    
  - Alpha<sub>v</sub> (α<sub>v</sub>): is the second step size that defines how much the second weight vector `v` is 
    updated at each time step.
- Beta (β): is the parameter used by the ETDβ algorithm that defines how much the product of importance sampling ratios
from the past affects the current update.
- Zeta (ζ): is only used in the ABTD algorithm. It is similar to the bootstrapping parameter of other algorithms.
- TDRCBeta (TDRCβ): is the regularization parameter of the TDRC algorithms. This parameter is often set to 1.

#### Variable naming conventions
- **w**: is the main weight vector being learned<sup>1</sup>.
- **v**: is the secondary weight vector learned by Gradient-TD algorithms<sup>1</sup>.
- delta (𝛿): is a numpy array or scalar representing the td-error, which in the full bootstrapping case, is equal to the
  reward plus the value of the next state, minus the value of the current state.
  delta is a scalar in the case that one policy is learned and is a numpy array in the case that multiple target 
  policies are learned. 
- s: is the current state (scalar).
- s_p: is the next state (scalar).
- r: is the reward (numpy array or scalar, similar to delta).
- **z**: is the eligibility trace vector (a matrix in the case where multiple target policies are learned).

> 1: a matrix in the case of multiple target policies.
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
