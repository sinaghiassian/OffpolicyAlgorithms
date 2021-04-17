<p align="center">
    ![Logo of RLAI](https://github.com/sinaghiassian/OffpolicyAlgorithms/Assets/rlai.png)
</p>

<h2 align=center>Off-policy Prediction Learning Algorithms</h2>
This repository ... 


<p align="center">
    ![Logo of RLAI](https://github.com/sinaghiassian/OffpolicyAlgorithmsAssets/fourRoomGridWorld.gif)
    ![Logo of RLAI](https://github.com/sinaghiassian/OffpolicyAlgorithmsAssets/fourRoomGridWorld.gif)
    ![Logo of RLAI](https://github.com/sinaghiassian/OffpolicyAlgorithmsAssets/fourRoomGridWorld.gif)
</p>
## Algorithms
- [Off-policy TD](#td)
- [GTD](#gtd)
- [Emphatic TD](#Emphatic_TD)

<hr>

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
    

## Environment