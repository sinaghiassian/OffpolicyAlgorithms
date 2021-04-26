<p align="center">
    <img width="100" src="/Assets/rlai.png" />
</p>
<br>
<div align="center">
  :steam_locomotive::train::train::train::train::train:
</div>
<h2 align=center>Off-policy Prediction Learning Algorithms</h2>

This repository includes the code for the "empirical off-policy" paper.
<br>


<p align="center">
    <img src="/Assets/FourRoomGridWorld.gif" />
    <img src="/Assets/chain.gif" />
</p>
<p align="center">
    <img src="/Assets/plots.png" />
</p>

## Table of Contents
- **[How to run the code](#how-to-run)**: [Learning.py](#learning.py), [Job Buidler](#job_builder)
- **[Algorithms](#algorithms)**
    - **[Algorithm Glossary](#glossary)**
    - **TD**: [Off-policy TD](#td)
    - **Gradient-TD family**   : [GTD](#gtd) , [GTD2](#gtd2), [HTD](#htd), [PGTD2](#pgdt2), [TDRC](#tdrc)
    - **Emphatic-TD family**   : [Emphatic TD](#etd), [Emphatic TDβ](#etdb)  
    - **Variable-λ family**    : [TB](#tb), [Vtrace](#vtrace), [ABTD](#abtd)
    - **Least squared family** : [LSTD](#lstd), [LSETD](#lsetd)
- **[Environments](#environment)** :  [Chain](#chain), [Four Room Grid World](#four_room_grid_world)
- **[Tasks](#tasks)** : [Collision](#collision), [Hallway proximity](#hallway_proximity), 
  [High variance hallway proximity](#highvar_hallway_proximity)



<a name='how-to-run'></a>
## How to Run the Code
The code can be run in two different ways.
One way is through `learning.py` that can be used to run small experiments on a local computer.
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
Algorithms are used to find a weight vector, [**w**](#var_w), such that the dot product of [**w**](#var_w) and the feature vector, 
approximates the value function. 

<a name='glossary'></a>
### Algorithm Glossary
Here, we briefly explain all the symbols and variables names that we use in our implementation.

#### meta-parameters
- Common parameters of all algorithms:
  - **alpha (α)**: is the step size that defines how much the weight vector [**w**](#var_w) is updated at each time step.
  - **lambda (λ)**: is the bootstrapping parameter.
- Common parameters of Gradient-TD algorithms:    
  - **alpha_v (α<sub>v</sub>)**: is the second step size that defines how much the second weight vector [**v**](#var_v) is 
    updated at each time step.
- **beta (β)**: is the parameter used by the [**ETDβ**](#etdb) algorithm that defines how much the product of importance sampling ratios
from the past affects the current update.
- **tdrc_beta (tdrc<sub>β</sub>)**: is the regularization parameter of the [**TDRC**](#tdrc) algorithms. This parameter is often set to 1.  
- **zeta (ζ)**: is only used in the [**ABTD**](#abtd) algorithm. It is similar to the bootstrapping parameter of other algorithms.

#### Algorithms variables
<a name='var_w'></a>
- **w**: is the main weight vector being learned<sup>1</sup>. ```init: w=0```.
<a name='var_v'></a>
- **v**: is the secondary weight vector learned by Gradient-TD algorithms<sup>1</sup>.  ```init: v=0```.
<a name='var_z'></a>
- **z**: is the eligibility trace vector<sup>1</sup>.  ```init: z=0```.
<a name='var_zb'></a>
- **z<sub>b</sub>**: is the extra eligibility trace vector used by [**HTD**](#htd)<sup>1</sup>.  ```init: z_b=0```.
<a name='var_delta'></a>
- delta (𝛿): is the td-error, which in the full bootstrapping case, is equal to the reward plus the value of the next 
  state minus the value of the current state<sup>2</sup>.
<a name='var_s'></a>
- s: is the current state (scalar).
<a name='var_x'></a>
- **x**: is the feature vector of the current state<sup>2</sup>.
<a name='var_s_p'></a>
- s_p: is the next state (scalar).
<a name='var_x_p'></a>
- **x_p**: is the feature vector of the next state<sup>2</sup>. 
<a name='var_r'></a>
- r: is the reward<sup>2</sup>.
<a name='var_rho'></a>
- rho (ρ): is the importance sampling ratio, which is equal to the probability of taking an action under the target policy
  divided by the probability of taking the same action under the behavior policy<sup>2</sup>.
<a name='var_oldrho'></a>
- old_rho (oldρ): is the importance sampling ratio at the previous time step<sup>2</sup>.
<a name='var_pi'></a>
- pi (π): is the probability of taking an action under the target policy at the current time step<sup>2</sup>.
<a name='var_oldpi'></a>
- old_pi (oldπ): is the probability of taking an action under the target policy in the previous time step. The variable
  π itself is the probability of taking action under the target policy at the current time step<sup>2</sup>.
<a name='var_F'></a>
- F : is the follow-on trace used by Emphatic-TD algorithms ???link??? <sup>2</sup>.
<a name='var_m'></a>
- m : is the emphasis used by Emphatic-TD algorithms ???link??? <sup>2</sup>.
<a name='var_nu'></a>
- nu (ν): Variable used by the ABQ/ABTD algorithm. ??more explanation??
<a name='var_si'></a>
- xi (ψ): Variable used by the ABQ/ABTD algorithm. ??more explanation??
<a name='var_mu'></a>
- mu (μ): is the probability of taking action under the behavior policy at the current time step<sup>2</sup>.
<a name='var_oldmu'></a>
- old_mu (oldμ): is the probability of taking an action under the target policy at the previous time step<sup>2</sup>.
- gamma (γ): is the discount factor parameter.

> <sub>1: a matrix in the case of multiple target policies.</sub> </br>
> <sub>2: numpy array in the case that multiple target policies are learned.</sub>

<a name='td'></a>
### Off-policy TD

**Paper** [Off-Policy Temporal-Difference Learning with Function Approximation](
https://www.cs.mcgill.ca/~dprecup/publications/PSD-01.pdf)<br>
**Authors** Doina Precup, Richard S. Sutton, Sanjoy Dasgupta<br>

```python
delta = r + gamma * np.dot(w, x_p) - np.dot(w, x)
z = rho * (gamma * lmbda * z + x)
w += alpha * delta * z
```

### Gradient-TD algorithms
<a name='gtd'></a>
#### GTD/TDC

**Paper** [Off-Policy Temporal-Difference Learning with Function Approximation](
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.160.6170&rep=rep1&type=pdf)<br>
**Authors** Richard S. Sutton, Hamid Reza Maei, Doina Precup, Shalabh Bhatnagar, David Silver, Csaba Szepesvàri,
Eric Wiewiora<br>

```python
delta = r + gamma * np.dot(w, x_p) - np.dot(w, x)
z = rho * (gamma * lmbda * z + x)
w += alpha * (delta * z - gamma * (1 - lmbda) * np.dot(z, v) * x_p)
v += alpha_v * (delta * z - np.dot(x, v) * x)
```

<a name='gtd2'></a>
#### GTD2

**Paper** [Off-Policy Temporal-Difference Learning with Function Approximation](
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.160.6170&rep=rep1&type=pdf)<br>
**Authors** Richard S. Sutton, Hamid Reza Maei, Doina Precup, Shalabh Bhatnagar, David Silver, Csaba Szepesvàri,
Eric Wiewiora<br>

```python
delta = r + gamma * np.dot(w, x_p) - np.dot(w, x)
z = rho * (gamma * lmbda * z + x)
w += alpha * (np.dot(x, v) * x - gamma * (1 - lmbda) * np.dot(z, v) * x_p)
v += alpha_v * (delta * z - np.dot(x, v) * x)
```

<a name='htd'></a>
#### HTD

**Paper** [Investigating Practical Linear Temporal Difference Learning](
https://arxiv.org/pdf/1602.08771.pdf)<br>
**Authors** Adam White, Martha White<br>

```python
delta = r + gamma * np.dot(w, x_p) - np.dot(w, x)
z = rho * (gamma * lmbda * z + x)
z_b = gamma * lmbda * z_b + x
w += alpha * ((delta * z) + (x - gamma * x_p) * np.dot((z - z_b), v))
v += alpha_v * ((delta * z) - (x - gamma * x_p) * np.dot(v, z_b))
```

<a name='pgtd2'></a>
#### Proximal GTD2

**Paper** [Proximal Gradient Temporal Difference Learning: Stable Reinforcement Learning with Polynomial Sample Complexity](
https://arxiv.org/pdf/2006.03976.pdf)<br>
**Authors** Bo Liu, Ian Gemp, Mohammad Ghavamzadeh, Ji Liu, Sridhar Mahadevan, Marek Petrik<br>

```python
delta = r + gamma * np.dot(w, x_p) - np.dot(w, x)
z = rho * (gamma * lmbda * z + x)
v_mid = v + alpha_v * (delta * z - np.dot(x, v) * x)
w_mid = w + alpha * (np.dot(x, v) * x - (1 - lmbda) * gamma * np.dot(z, v) * x_p)
delta_mid = r + gamma * np.dot(w_mid, x_p) - np.dot(w_mid, x)
w += alpha * (np.dot(x, v_mid) * x - gamma * (1 - lmbda) * np.dot(z, v_mid) * x_p)
v += alpha_v * (delta_mid * z - np.dot(x, v_mid) * x)
```

<a name='tdrc'></a>
#### TDRC

**Paper** [Gradient Temporal-Difference Learning with Regularized Corrections](
http://proceedings.mlr.press/v119/ghiassian20a/ghiassian20a.pdf)<br>
**Authors** Sina Ghiassian, Andrew Patterson, Shivam Garg, Dhawal Gupta, Adam White, Martha White <br>

```python
delta = r + gamma * np.dot(w, x_p) - np.dot(w, x)
z = rho * (gamma * lmbda * z + x)
w += alpha * (delta * z - gamma * (1 - lmbda) * np.dot(z, v) * x_p)
v += alpha_v * (delta * z - np.dot(x, v) * x) - alpha_v * tdrc_beta * v
```

### Emphatic-TD algorithms

<a name='etd'></a>
#### Emphatic TD

**Paper** [An Emphatic Approach to the Problem of Off-policy Temporal-Difference Learning](
https://jmlr.org/papers/volume17/14-488/14-488.pdf)<br>
**Authors** Richard S. Sutton, A. Rupam Mahmood, Martha White<br>

```python
delta = r + gamma * np.dot(w, x_p) - np.dot(w, x)
z = rho * (gamma * lmbda * z + x)
F = gamma * old_rho * F + 1
m = lmbda * 1 + (1 - lmbda) * F
z = rho * (x * m + gamma * lmbda * z)
w += alpha * delta * z
```

<a name='etdb'></a>
#### Emphatic TDβ

**Paper** [An Emphatic Approach to the Problem of Off-policy Temporal-Difference Learning](
https://jmlr.org/papers/volume17/14-488/14-488.pdf)<br>
**Authors** Richard S. Sutton, A. Rupam Mahmood, Martha White<br>

```python
delta = r + gamma * np.dot(w, x_p) - np.dot(w, x)
z = rho * (gamma * lmbda * z + x)
F = beta * old_rho * F + 1
m = lmbda * 1 + (1 - lmbda) * F
z = rho * (x * m + gamma * lmbda * z)
w += alpha * delta * z
```


### Variable-λ algorithms

<a name='tb'></a>
#### Tree backup/ Tree backup for prediction

**Paper** [Eligibility Traces for Off-Policy Policy Evaluation](
https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&=&context=cs_faculty_pubs&=&sei-redir=1&referer=https%253A%252F%252Fscholar.google.com%252Fscholar%253Fhl%253Den%2526as_sdt%253D0%25252C5%2526q%253Dtree%252Bbackup%252Balgorithm%252Bdoina%252Bprecup%2526btnG%253D#search=%22tree%20backup%20algorithm%20doina%20precup%22)<br>
**Authors** Doina Precup, Richard S. Sutton, Satinder Singh<br>

The algorithm pseudo-code described below is the prediction variant of the original Tree backup algorithm proposed by 
Precup, Sutton, and Singh (2000). The prediction variant of the algorithm used here is first derived in the current paper.
```python
delta = rho * (r + gamma * np.dot(w, x_p) - np.dot(w, x))
z = gamma * lmbda * old_pi * z + x
w = w + alpha * delta * z
```

<a name='vtrace'></a>
#### Vtrace

**Paper** [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](
https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&=&context=cs_faculty_pubs&=&sei-redir=1&referer=https%253A%252F%252Fscholar.google.com%252Fscholar%253Fhl%253Den%2526as_sdt%253D0%25252C5%2526q%253Dtree%252Bbackup%252Balgorithm%252Bdoina%252Bprecup%2526btnG%253D#search=%22tree%20backup%20algorithm%20doina%20precup%22)<br>
**Authors** Lasse Espeholt,  Hubert Soyer,  Remi Munos,  Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu <br>

```python
delta = r + gamma * np.dot(w, x_p) - np.dot(w, x)
z = min(1, rho) * (gamma * lmbda * z + x)
w += alpha * delta * z
```

<a name='abtd'></a>
#### ABQ/ABTD

**Paper** [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](
https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&=&context=cs_faculty_pubs&=&sei-redir=1&referer=https%253A%252F%252Fscholar.google.com%252Fscholar%253Fhl%253Den%2526as_sdt%253D0%25252C5%2526q%253Dtree%252Bbackup%252Balgorithm%252Bdoina%252Bprecup%2526btnG%253D#search=%22tree%20backup%20algorithm%20doina%20precup%22)<br>
**Authors** Lasse Espeholt,  Hubert Soyer,  Remi Munos,  Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu <br>

The algorithm pseudo-code described below is the prediction variant of the original Tree backup algorithm proposed by 
Mahmood, Sutton, and Yu (2017). The prediction variant of the algorithm used here is first derived in the current paper.
```python
delta = rho * (r + gamma * np.dot(w, x_p) - np.dot(w, x))
nu = min(si, 1.0 / max(pi, mu))
z = x + gamma * old_nu * old_pi * z
w += alpha * delta * z
```


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
