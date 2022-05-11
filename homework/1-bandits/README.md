# Assignment 1: Bandits

This readme file gives more information on the first assignment of the Reinforcement Learning (RL) course. It also includes the solution of Lennert Bontinck.

## Table of contents

- [Contact information](#contact-information)
- [Assignment](#assignment)
   - [About the assignment](#about-the-assignment)
   - [About the bandit](#about-the-bandit)
- [Challenge of the homework](#challenge-of-the-homework)
- [Discussion on the solution](#discussion-on-the-solution)
   - [Upper Confidence Bound](#upper-confidence-bound)
   - [Implementation](#implementation)
   - [Results](#results)
   - [Conclusion](#conclusion)
   
- [Running the code](#running-the-code)

<hr>


## Contact information

| Name             | Student ID | VUB mail                                                  | Personal mail                                               |
| ---------------- | ---------- | --------------------------------------------------------- | ----------------------------------------------------------- |
| Lennert Bontinck | 0568702    | [lennert.bontinck@vub.be](mailto:lennert.bontinck@vub.be) | [info@lennertbontinck.com](mailto:info@lennertbontinck.com) |

<hr>


## Assignment

### About the assignment

This directory contains the implementation of a bandit (bandit.py) and the skeleton of a bandit learning program (main.py). The goal of this assignment is to implement learning rules in main.py so that the agent learns the bandit. The second goal is to learn fast: having the highest learning curve, or lowest cumulative regret, among the students doing this exercise, gives bonus points.

### About the bandit

For the agent, the bandit is just a set of arms that give outcomes. You are not allowed to use any additional knowledge you have on the bandit in your solution.
Your algorithm must be general. However, for information, here are details about the bandit:

- The bandit represents what happens in a big population of potentially-sick individuals when vaccination strategies are implemented.
   - Pulling one arm maps, in the real world, to:
      - The infection of a couple of people
      - The assignation of vaccines to specific age groups
   - The reward given by the arm is computed from the total amount of people who became sick in 2 or 3 years.
   - You can see why sample efficiency is important here: in the real world, those kinds of simulations (pulling a single arm) take about 1 hour on a supercomputer.
      - For this assignment, thousands of outcomes for every arm have been generated and kept in the distributions/ directory, so that your computer does not have to run hot.
- The outcome of each arm is bi-modal.
   - Remember, you cannot use this information to tune your algorithm. 
   - This means that either the epidemic dies out (with a small probability), or it takes over the population.
   - Some fancy work done at the AI lab uses this hypothesis to learn the bandit a bit faster.


<hr>


## Challenge of the homework

K-armed bandits are a special type of RL problem where there is only one state. Thus, the K-armed bandit problem is a type of non-associative single state RL. In a K-armed bandit setting, an agent is connected to one or more bandits who themselves have arms that can be pulled. Pulling such an arm will give a reward to the agent. The goal of the agent is to optimize the total received reward. Domain knowledge, i.e. knowledge about the arms of the bandits, can aid in creating an optimal strategy for the agent. However, for our solution, we are asked to work in a general manner and thus to not use any knowledge about the bandit or its arms other than the very basic information of how many arms it has and wanting to be sample efficient. 

As we have seen, solving a K-armed bandits problem is mainly finding a balance between exploration and exploitation of the agent's strategy for pulling arms. The word balance is important here, as we don't directly solve the exploration/exploitation problem, nor does it represent the general RL exploration/exploitation problem. In general, we know that:

- A random agent will have excellent exploration but won't use the experience it has gotten and thus it won't optimize itself to pick the best possible arm w.r.t. the expected reward.
- Greedy agents are agents that opt to pull the arm with the highest action value. The action value is an agent's determined expected reward from using an arm based on previous experience. In the most simple form, this can be the previous reward or an average of all previous rewards. In this setting, agents do optimize for what they believe is the most optimal reward, but their approximations for the expected reward can be wrong resulting in them picking a suboptimal solution.
- Thus, an agent needs a combination of being greedy whilst still allowing for exploration. Amongst the simplest examples is an epsilon greedy approach. It's a combination of a random agent and a greedy agent, where the agent is greedy except for a certain percentage of the time where it acts as a random agent to force the agent to remain explorative.

However, epsilon greedy using a fixed epsilon can be suboptimal, especially if the goal is stationary. Indeed, if we have a specific certainty for the expected value of all arms, why would one still explore as much as was the case in the beginning? To tackle this issue, many variants exist such as a time decaying epsilon greedy. Starting from other initial values, such as an optimistic very high value for each arm, can also improve the performance of these approaches.

When working with non-stationary targets, we want to alter the agent's expected reward for an arm so that more recent samples have a higher impact, as to base our beliefs of the most optimal strategy on more recent samples rather than later ones. One such example is using a recency-weighted average for updating the average reward of an arm. However, if noise is high and we focus the expected reward on new samples heavily, it can cause heavy fluctuations of the expected rewards and thus result in suboptimal solutions.

Other interesting approaches include SoftMax, where a temperature amongst other variables controls the exploration. However, deciding on a good value for such a temperature requires domain knowledge which is often not available. Other such interesting approaches include gradient bandits and agents working with a regret where the received reward is compared with a theoretical best. But again, this would require domain knowledge.

Finally, Upper Confidence Bound (UCB) based agents are known to be very effective in solving bandit problems. It was discussed during the lecture that if you want to solve bandit problems, using UCB is a good approach.


<hr>


## Discussion on the solution

Since we are asked to create a general algorithm that doesn't use bandit specific information, we rule out the use of approaches such as SoftMax that do require domain knowledge to choose fit parameters, i.e. the temperature. Whilst we have the bandit function `opt()` for retrieving the optimal reward and thus are capable of using regret based approaches, we believe such approaches are also not general and thus we won't use this information in our solution.



### Upper Confidence Bound

Our solution is based on the discussed Upper Confidence Bound (UCB) approach. Contrary to epsilon-greedy where an arm is chosen at random during the exploration phase, UCB takes into account how it got the estimated reward for an arm to influence which arm to pull next.

Intuitively, the idea behind this is as follows. Imagine we have two arms where one has an estimated reward of 28 and the other of 31:

- In case both arms are pulled sufficiently already (e.g. thousands of times)
   - Epsilon greedy would pick the arm with an estimated reward of 31 in the greedy occasion, which is ideal.
   - Epsilon greedy would pick a random arm a certain percentage of the time, which in the case of a fixed epsilon is likely unneeded as the confidence of a good estimation is already sufficient and the agent can be more greedy than initially.
- In case the arm of 28 is only pulled once and the arm of 31 is pulled many times already
   - Epsilon greedy would have to wait on the random selection to improve the estimated reward of the arm that was only pulled once, which might make it turn out that that arm is better than the other one.
- Thus, as humans would intuitively also do, we want to base our selection not only on the expected reward but also on how certain we are of those expected rewards. This is what UCB aims to achieve. 

UCB builds on the idea described above by using an action selection function as follows:
$$
A_t = argmax_a (Q_t(a) + c \sqrt{\frac{ln(t)}{N_t(a)}})
\\ \\
A_t = \text{Action taken at timestep } t
\\ \\
Q_t = \text{Action value estimate, favours exploitation}
\\ \\
c \sqrt{\frac{ln(t)}{N_t(a)}} = \text{Part for clever exploration based on Hoeffding's inequality}
\\ \\
c = \text{Weighting factor for controlling exploration behaviour}
\\ \\
ln(t) = \text{natural log of the current time step}
\\ \\
N_t = \text{number of times action a was performed so far}
$$

It is noted that [the excellent explanatory video by ritvikmath](https://youtu.be/FgmMK6RPU1c) was used to gain further insight on UCB. It thought us that one area where UCB is known to struggle is when there are a lot of arms and a small number of timesteps. In such a scenario purely greedy algorithms can outperform UCB, which might take too much time in trying to explore and stabilise the estimated reward of all arms first before becoming greedy. Having 32 arms and taking into account that an arm pull would take around an hour on a super computer, it is not unreasonable to think this could form a problem. However, we still believe as a general approach UCB is better than a more greedy approach which might have better results on limited timesteps but is expected to become worse again once timesteps increases. It is also possible to lower `c` in the equation above to increase the greediness of the algorithm.



### Implementation

We created an agent class of which an object can be created as follows:

```python
agent = Agent(
    weighting_factor_c= 0.03, # Weighting factor of exploration in UCB (default 0.05)
    use_recency_weighted_average= False, # Whether to use recency-weighted average for estimated reward (default False - regular avg)
    recency_weight = 0.1, # Weighting factor for recency when using recency-weighted average (default 0.1)
    initial_value= 0 # Initial estimated reward for actions (default None - uses bandit.opt() for init)
)
```

Since we know UCB can perform poorly on settings where there are few samples with a high amount of arms due to it exploring too much, we created an if in the main function that checks for the environment as follows:

```python
# Depending on the situation use a more exploratory or more greedy approach
if ((num_arms * 20) > timesteps):
    # Few samples compared to amount of arms, be more greedy
    print("Few samples, choosing more greedy agent")
    agent = Agent(
        weighting_factor_c= 0.03, # Default 0.05
        use_recency_weighted_average= False, # Default False
        initial_value= 0 # Default None for bandit specified opt
    )
else:
    # Enough samples, be less greedy
    print("Enough samples, choosing more exploratory agent")
    agent = Agent(
        weighting_factor_c= 0.075, # Default 0.05
        use_recency_weighted_average= False, # Default False
        initial_value= None # Default None for bandit specified opt
    )
```

As you can see, in the case there are few samples we have an agent that performs more greedy due to a lower weighting factor C for the UCB and initialisation of 0. On the contrary, when there are many samples, we allow for more exploration by increasing the weighting factor C for the UCB and using the optimal value as the initial value. Neither of the two uses the recency weighted average as we believe it to be unnecessary in this setting.



### Results

We consider three experiments of which each was performed 10 times. The results of these experiments are available in the `experiment_results.xlsx` Excel file:

- 120 samples, corresponding to 5 days of supercomputer time if 1 action corresponds with 1 hour (24 * 5 = 120)
   - The average total reward is: 30.4119
   - The average reward per action is: 0.2534
   - 8/10 times arm 0 was most chosen, the other 2 this was arm 1
   - When arm 0 is most chosen, it is on average chosen 14.1667% of the time
- 1000 samples, to test the performance of a less greedy agent on a relative small sample size
   - The average total reward is: 254.0991
   - The average reward per action is: 0.2540
   - 8/10 times arm 0 was most chosen, the other 2 this was arm 1 and 7
   - When arm 0 is most chosen, it is on average chosen 14.4750% of the time
- 10000 samples, to test if the agent becomes better over time
   - The average total reward is: 2679.4576
   - The average reward per action is: 0.2679
   - 0 was the most chosen arm in all experiments
   - Arm 0 is on average chosen 72.80% of the time


| **Amount of samples** | **Average total reward** | **Average action reward** | **Most choosen arm** | **Average arm 0 choice rate** |
| --------------------- | ------------------------ | ------------------------- | -------------------- | ----------------------------- |
| 120                   | 30.4119                  | 0.2534                    | 0 (8/10)             | 14.1667 %                     |
| 1000                  | 254.0991                 | 0.2540                    | 0 (8/10)             | 14.4750 %                     |
| 10000                 | 2679.4576                | 0.2679                    | 0 (10/10)            | 72.80 %                       |




### Conclusion

We created an agent that is capable of using the Upper Confidence Bound (UCB) approach for learning the K-armed bandit game. We didn't use any task specific information and determine upon initialisation if we want a more or less greedy agent based on the number of samples available and the number of arms available. The latter is done since it is known that UCB can be too exploratory if the total allowed samples are relatively low. Our results all point to arm 0 being the best arm and as samples increase the average reward also increases, as does the average choice for arm 0. The parameters of the agent can be set upon the initialisation of the agent and a recency-weighted estimated reward variant is possible although not used for our experiments.


<hr>


## Running the code

An Anaconda environment based on Python 3.8.10 was used for this homework. More information on this environment can be found in the [Anaconda environment documentation for the homework](../../documentation/README.md).

With the Anaconda Python environment installed as specified above, running the code is as simple as calling the `main.py` file and providing an optional argument that specifies the number of timesteps to take (defaults to 120). An example call of the `main.py` file is as follows:

```bash
# Call the main.py file with an optional parameter specifying the number of timesteps to take
## Example of filled in call:
## & C:/Users/Lennert/.conda/envs/rl-homework/python.exe c:/fast_files/GitHub/VUB-RL/homework/1-bandits/main.py 120
& & {path/to/conda}/.conda/envs/rl-homework/python.exe {path/to/github}/VUB-RL/homework/1-bandits/main.py 120
```



