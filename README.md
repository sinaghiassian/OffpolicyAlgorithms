
<div align="center">
  :steam_locomotive::train::train::train::train::train:
</div>
<h2 align=center>An Empirical Comparison of Off-policy Prediction Learning Algorithms on the Collision Task</h2>



<p align="center">
    <img src="/Assets/chain.gif" />
</p>
<p align="center">
    <img src="/Assets/plots.png" />
</p>

## Table of Contents
- **[Specification of Dependencies](#specifications)**
- **[Algorithms](#algorithms)**
    - **TD**: [Off-policy TD](#td)
    - **Gradient-TD family**   : [GTD](#gtd) , [GTD2](#gtd2), [HTD](#htd), [PGTD2](#pgdt2), [TDRC](#tdrc)
    - **Emphatic-TD family**   : [Emphatic TD](#etd), [Emphatic TDβ](#etdb)  
    - **Variable-λ family**    : [TB](#tb), [Vtrace](#vtrace), [ABTD](#abtd)
    - **[Algorithm Glossary](#glossary)**
- **[Environments](#environment)** :  [Chain](#chain), [Four Room Grid World](#four_room_grid_world)
- **[How to run the code](#how-to-run)**: [Learning.py](#learning.py), [Job Buidler](#job_builder)
- **[Plotting the results](#Plot-results)**

<a name='specifications'></a>
## Specification of Dependencies
This code requires python 3.5 or above. Packages that are required for running the code are all in the `requirements.txt`
file. To install these dependencies, run the following command if your pip is set to `python3.x`:
```text
pip install requirements.txt
```
otherwise, run:
```text
pip3 install requirements.txt
```




<a name='algorithms'></a>
## Algorithms
Algorithms are used to find a weight vector, [**w**](#var_w), such that the dot product of [**w**](#var_w) and the feature vector, 
approximates the value function. 


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
#### Vtrace (simplified)

**Paper** [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures]
(http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf)<br>
**Authors** Lasse Espeholt,  Hubert Soyer,  Remi Munos,  Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu <br>

```python
delta = r + gamma * np.dot(w, x_p) - np.dot(w, x)
z = min(1, rho) * (gamma * lmbda * z + x)
w += alpha * delta * z
```

<a name='abtd'></a>
#### ABQ/ABTD

**Paper** [Multi-step Off-policy Learning Without Importance Sampling Ratios](
https://arxiv.org/pdf/1702.03006)<br>
**Authors** A. Rupam Mahmood, Huizhen Yu, Richard S. Sutton <br>

The algorithm pseudo-code described below is the prediction variant of the original Tree backup algorithm proposed by 
Mahmood, Sutton, and Yu (2017). The prediction variant of the algorithm used here is first derived in the current paper.
This algorithm first needs to compute the following:
```python
xi_zero = 1
xi_max = 2
xi = 2 * zeta * xi_zero + max(0, 2 * zeta - 1) * (xi_max - 2 * xi_zero)
```
`xi_zero` and `xi_max` are specifically computed here for the Collision problem. 
To see how these are computed for the task see the original paper referenced above.

```python
nu = min(xi, 1.0 / max(pi, mu))
delta = rho * (r + gamma * np.dot(w, x_p) - np.dot(w, x))
nu = min(xi, 1.0 / max(pi, mu))
z = x + gamma * old_nu * old_pi * z
w += alpha * delta * z
```

<a name='glossary'></a>
### Algorithm Glossary
Here, we briefly explain all the symbols and variables names that we use in our implementation.

#### meta-parameters
- Common parameters of all algorithms:
  - alpha (α): is the step size that defines how much the weight vector [**w**](#var_w) is updated at each time step.
  - lambda (λ): is the bootstrapping parameter.
- Common parameters of Gradient-TD algorithms:    
  - alpha_v (α<sub>v</sub>): is the second step size that defines how much the second weight vector [**v**](#var_v) is 
    updated at each time step.
- beta (β): is the parameter used by the [**ETDβ**](#etdb) algorithm that defines how much the product of importance sampling ratios
from the past affects the current update.
- tdrc_beta (tdrc<sub>β</sub>): is the regularization parameter of the [**TDRC**](#tdrc) algorithms. This parameter is often set to 1.  
- zeta (ζ): is only used in the [**ABTD**](#abtd) algorithm. It is similar to the bootstrapping parameter of other algorithms.

#### Algorithms variables
<a name='var_w'></a>
- **w**: is the main weight vector being learned. ```init: w=0```.
<a name='var_v'></a>
- **v**: is the secondary weight vector learned by Gradient-TD algorithms.  ```init: v=0```.
<a name='var_z'></a>
- **z**: is the eligibility trace vector.  ```init: z=0```.
<a name='var_zb'></a>
- **z<sub>b</sub>**: is the extra eligibility trace vector used by [**HTD**](#htd).  ```init: z_b=0```.
<a name='var_delta'></a>
- delta (𝛿): is the td-error, which in the full bootstrapping case, is equal to the reward plus the value of the next 
  state minus the value of the current state.
<a name='var_s'></a>
- s: is the current state (scalar).
<a name='var_x'></a>
- **x**: is the feature vector of the current state.
<a name='var_s_p'></a>
- s_p: is the next state (scalar).
<a name='var_x_p'></a>
- **x_p**: is the feature vector of the next state. 
<a name='var_r'></a>
- r: is the reward.
<a name='var_rho'></a>
- rho (ρ): is the importance sampling ratio, which is equal to the probability of taking an action under the target policy
  divided by the probability of taking the same action under the behavior policy.
<a name='var_oldrho'></a>
- old_rho (oldρ): is the importance sampling ratio at the previous time step.
<a name='var_pi'></a>
- pi (π): is the probability of taking an action under the target policy at the current time step.
<a name='var_oldpi'></a>
- old_pi (oldπ): is the probability of taking an action under the target policy in the previous time step. The variable
  π itself is the probability of taking action under the target policy at the current time step.
<a name='var_F'></a>
- F : is the follow-on trace used by [Emphatic-TD](#etd) algorithms.
<a name='var_m'></a>
- m : is the emphasis used by [Emphatic-TD](#etd) algorithms.
<a name='var_nu'></a>
- nu (ν): Variable used by the ABQ/ABTD algorithm. Please refer to the [original paper](https://arxiv.org/pdf/1702.03006) for explanation.
<a name='var_si'></a>
- xi (ψ): Variable used by the ABQ/ABTD algorithm. Please refer to the [original paper](https://arxiv.org/pdf/1702.03006) for explanation.
<a name='var_mu'></a>
- mu (μ): is the probability of taking action under the behavior policy at the current time step.
<a name='var_oldmu'></a>
- old_mu (oldμ): is the probability of taking an action under the target policy at the previous time step.
- gamma (γ): is the discount factor parameter.


<a name='environment'></a>
## Environment
At the heart of an environment is an MDP.
The MDP defines the states, actions, rewards, transition probability matrix, and the discount factor.

<a name="chain_env"></a>
### Chain Environment and the Collision Task
<br>
<p align="center">
    <img width="800" src="/Assets/eight_state_collision.png" />
</p>
<br>
An MDP with eight states is at the heart of the task.
The agent starts in one of the four leftmost states with equal probability.
One action in available in the four leftmost states: forward. Two actions are available in the four rightmost states: 
forward and turn. By taking the forward action, the agent transitions one state to the right and by taking the turn 
action, it moves away from the wall and transitions to one of the four leftmost states equiprobably. Rewards are all 
zero except for taking forward in state 8 for which a +1 is emitted. Termination function (discount factor) returns
0.9 for all transitions except for taking turn in any state or taking forward in state 8, for which the termination
function returns zero.

```python
env = Chain()
env.reset() # returns to one of the four leftmost states with equal probability.
for step in range(1, 1000):
    action = np.random.randint(0, 2) #  forward=0, turn=1
    sp, r, is_wall = env.step(action=action)
    if is_wall:
        env.reset()
```

We applied eleven algorithms to the Collision task: Off-policy TD(λ), GTD(λ), GTD2(λ), HTD(λ), Proximal GTD2(λ), TDRC(λ)
, ETD(λ), ETD(λ,β), Tree Backup(λ), Vtrace(λ), ABTD(ζ). The target policy was π(forward|·) = 1.0. The behavior policy 
was b(forward|·) = 1.0 for the four leftmost states and b(forward|·) = 0.5, b(retreat|·) = 0.5 for the four rightmost 
states. Each algorithm was applied to the task with a range of parameters. We refer to an algorithm with a specific 
parameter setting as an instance of that algorithm. Each algorithm instance was applied to the Collision task for 
20,000 time steps, which we call a run. We repeated the 20,000 time steps for fifty runs. All instances of all 
algorithms experienced the same fifty trajectories.

Linear function approximation was used to approximate the true value function. Each state was represented by a six 
dimensional binary feature vector. The feature representation of each state had exactly three zeros and three ones. 
The locations of the zeros and ones were selected randomly. This was repeated once at the beginning of each run, 
meaning that the representation for each run is most probably different from other runs. At the beginning of each run 
we set **w**<sub>0</sub> = **0** and thus the error would be the same for all algorithms at the beginning of the runs.

#### Feature representation
The feature representation for the collision task is an array of size `8, 6, 50`, where 8 corresponds to the number of 
states, 6 correponds to the number of features for each state, and 50 corresponds to the number of runs.
The feature representations used for the set of results presented here and in the paper is saved in:
```
Resources/EightStateCollision/feature_rep.npy
```
Note that the feature representaiton for each run is different in the Collision task.
For example, the feature representation for the first run is:
```
array([[0., 0., 1., 0., 1., 1.],
       [1., 1., 1., 0., 0., 0.],
       [0., 1., 1., 0., 0., 1.],
       [1., 0., 1., 1., 0., 0.],
       [1., 1., 0., 0., 1., 0.],
       [0., 1., 1., 1., 0., 0.],
       [1., 1., 0., 0., 0., 1.],
       [1., 0., 1., 0., 0., 1.]])
```

#### State distribution induced by the behavior policy
To compute an approximation of the mean squared value error at each time step, weighting induced by the behavior policy
was approximated by following the behavior policy for 20,000,000 time step and computing the fraction of time spent in
each state. The resulting distribution is saved in:
```
Resources/EightStateCollision/d_mu.npy
```
`d_mu.npy` is a one dimensional numpy array of size `8`:
```
array([0.05715078, 0.1142799 , 0.17142456, 0.22856842, 0.22856842, 0.11428067, 0.05715311, 0.02857415])
```

#### True state values
To compute an approximation of the mean squared value error at each time step, we need the true state values.
Luckily, for the Collision task, these values are easy to compute.
We computed these true values by following the target policy from each state to the wall once.
The resulting values are saved in:
```
Resources/EightStateCollision/state_values.npy
```
`state_values.npy` is a one dimensional numpy array of size `8`:
```
array([0.4782969, 0.531441, 0.59049, 0.6561, 0.729, 0.81, 0.9, 1])
```



<a name='how-to-run'></a>
## How to Run the Code
The code can be run in two different ways.
One way is through `learning.py` that can be used to run small experiments on a local computer.
The other way is through the files inside the Job directory. 
We explain each of these approaches below by means of an example.

### Running on Your Local Machine
Let's take the following example: applying Off-policy TD(λ) to the Collision task.
There are multiple ways for doing this.
The first way is to open a terminal and go into the root directory of the code and run `Learning.py` with proper parameters:
```
python3 Learning.py --algorithm TD --task EightStateCollision --num_of_runs 50 --num_steps --environment Chain
--save_value_function Ture --alpha 0.01 --lmbda 0.9
```
In case any of the parameters are not specified, a default value will be used.
The default value is set in the `Job` directory, inside the `JobBuilder.py` file.
This means, the code, can alternatively be run, by setting all the necessary values that an algorithm needs at the top of the `JobBuilder.py` file.
Note that not all parameters specified in the `default_params` dict are required for all algorithms. For example, the `tdrc_beta` parameter is only
required to be set for the TDRC(λ) algorithms.
Once the variables inside the `default_params` dictionary, the code can be run:
```
python3 Learning.py
```
Or one can choose to specify some parameters in the `default_params` dictionary and specify the rest as command line argumets 
like the following:
```
python3 Learning.py --algorithm TD --task EightStateCollision --alpha 0.01
```

### Running on Servers with Slurm Workload Managers
When parameter sweeps are necessary, the code can be run on supercomputers. 
The current code supports running on servers that use slurm workload managers such as compute canada.
For exampole, to apply the TD algorithm to the Collision (EightStateCollision) task, with various parameters,
first you need to create a json file that specifies all the parameters that you would like to run, for example:
```json
{
  "agent": "TD",
  "environment": "Chain",
  "task": "EightStateCollision",
  "number_of_runs": 50,
  "number_of_steps": 20000,
  "sub_sample": 1,
  "meta_parameters": {
    "alpha": [
      0.000003814, 0.000007629, 0.000015258, 0.000030517, 0.000061035, 0.000122070, 0.000244140, 0.000488281,
      0.000976562, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0
    ],
    "lmbda": [
      0.1, 0.2, 0.3
    ]
  }
}
```
and then run `main.py` using python:
```
python3 main.py -f <path_to_the_json_file> -s <kind_of_submission>
```
where `kind_of_submission` refers to one of the two ways you can submit your code:
1) You can request an individual cpu for each of the algorithm instances, where an algorithm instance refers to an 
algorithm with specific parameters. To request an individual cpu, run the following command:
```
python3 main.py -f <path_to_the_json_file_or_dir> -s cpu
```
When running each algorithm instance on a single cpu, you need to specify the following parameters inside 
`Job/SubmitJobsTemplatesCedar.SL`:
```shell
#SBATCH --account=xxx
#SBATCH --time=00:15:58
#SBATCH --mem=3G
```
where `#SBATCH --account=xxx` requires the account you are using in place of `xxx`,
`#SBATCH --time=00:15:58` requires the time you want to request for each individual cpu,
and `#SBATCH --mem=xG` requires the amount of memory in place of x.

2) You can request a node, that we assume includes 40 cpus. If you request a node, the jobs you submit will run in 
parallel 40 at a time, and once one job is finished, the next one in line will start running.
This process continues until either all jobs are finished running, or you run out of the time you requested for that node.
```
python3 main.py -f <path_to_the_json_file_or_dir> -s node
```
When running the jobs on nodes, you need to specify the following parameters inside `Job/SubmitJobsTemplates.SL`:
```shell
#SBATCH --account=xxx
#SBATCH --time=11:58:59
#SBATCH --nodes=x
#SBATCH --ntasks-per-node=40
```
where `#SBATCH --account=xxx` requires the account you are using in place of `xxx`,
`#SBATCH --time=11:58:59` requires the time you want to request for each individual node, each of which includes 40 cpus in this case,
and `#SBATCH --nodes=x` requires the number of nodes you would like to request in place of x.
If you request more than one node, your jobs will be spread across nodes, 40 on each node, and once each job finishes, 
the next job in the queue will start running.
`#SBATCH --ntasks-per-node=xx` is the number of jobs you would like to run concurrently on a single node. In this case,
for example, we set it to 40.

If `path_to_the_json_file_or_dir` is a directory, then the code will walk into all the subdirectories, and submits jobs for
all the parameters in the json files that it finds inside those directories sequentially.
If `path_to_the_json_file_or_dir` is a file, then the code will submit jobs for all the parameters that it finds inside that 
single json file.
Note that you can create a new directory for each experiment that you would like to run, and create directories for each
of the algorithms you would like to run in that experiment.
For example, we created a directory called `FirstChain` inside the `Experiments` directory and created one directory
per algorithm inside the `FirstChain` directory for each of the algorithms and specified a json file in that directory.
It is worth noting that whatever parameter that is not specified in the json file will be read from the `default_params`
dictionary inside the `Job` directory inside the `JobBuilder.py` file.


<a name='Plot-results'></a>
## Plotting the results
The following table shows all the parameters that we tested in the experiments:
<p align="center">
    <img width="700" src="/Assets/parameters.png" />
</p>

We now explain how each figure in the paper can be reproduced.
All the figures of the paper can be reproduced using the `plot_data.py` file.

1) **Processing the data**: This script manipulates data in a way that it is ready to be plotted over step sizes and also such 
   that the data is ready to be plotted as learning curves averaged over runs.
   The `process_data` script also re-runs the algorithms with their best parameters to eliminate possible maximization 
   bias, as explained in the paper.
   This is a time consuming step. If you do not like to do this step, simply set:
    ```python
    PLOT_RERUN = False
    ```
    in `Plotting/plot_params.py` and the code will ignore the re-running steps.
    If you would like to eliminate the maximization bias, set:
    ```python
    PLOT_RERUN = True
    ```
   Finally, go to `plot_data.py` and set `func_to_run = 'process_data'`, and run the `plot_data.py` script.

2) **Plotting the learned value functions**: 
   Go to `plot_data`, and set `func_to_run = 'plot_value_functions'` to plot 
   the learned value functions for some of the runs, and set `func_to_run = plot_all_final_value_functions` to plot the 
   value function learned by the last time step of all of the runs in one plot.
   <p align="center">
   <img src="/Assets/value_functions.png" />
   </p>
   <br>

3) **Plotting the learning curves with specific parameter values**: 
   Go to `plot_data`, and set  `func_to_run = 'specific_learning_curves_full_bootstrap'`, and run the `plot_data.py` 
   script.
   <br></br>
   <p align="center">
   <img width="450" src="/Assets/specific_learning_curves.png" />
   </p>
   <br>
   
4) **Plotting the parameter studies for step size for all algorithms**: 
   Go to `plot_data`, and set  `func_to_run = 'collision_sensitivity_curves_for_many_lambdas'`, and run the script.
   <br></br>
   <p align="center">
   <img src="/Assets/sensitivity_curves_of_all_algs.png" />
   </p>
   <br>
   
5) **Plotting the parameter sensitivity study of Emphatic-TD algorithms**: 
   Go to `plot_data`, and set  `func_to_run = 'collision_emphatics_sensitivity_full_bootstrap'`, and run the script.
   <br></br>
   <p align="center">
   <img width="550" src="/Assets/Emphatics_sensitivity.png" />
   </p>
   <br>
   
6) **Plotting the parameter sensitivity study of Gradient-TD algorithms**: 
   Go to `plot_data`, and set  `func_to_run = 'collision_gradients_sensitivity_full_bootstrap'`, and run the script.
   <br></br>
   <p align="center">
   <img width="850" src="/Assets/Gradients_sensitivity.png" />
   </p>
   <br>
