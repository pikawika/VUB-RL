# Assignment 1: Bandits

This readme file gives more information on the first assignment of the Reinforcement Learning (RL) course. It also includes the solution of Lennert Bontinck.

## Table of contents

- [Contact information](#contact-information)
- [Assignment](#assignment)
   - [About the assignment](#about-the-assignment)
   - [About the bandit](#about-the-bandit)
- [Challenge of the homework](#challenge-of-the-homework)
- [Discussion on the solution](#discussion-on-the-solution)
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

During the introduction of the RL course, we saw that agents are in specific states which can change when taking actions. However, K-armed bandits are a special type of RL problem where there is only one state. In short, the K-armed bandit problem is a type of non-associative single state RL. An agent can pull one of the arms available and will get a result in return. The goal is to optimize this reward.

As we have seen, solving a K-armed bandits problem is mainly finding a balance between exploration and exploitation of the agent's strategy for pulling arms. The word balance is important here, as we don't solve the problem, nor does it represent the general RL exploration/exploitation problem. We know that:

- A random agent will have excellent exploration but won't learn anything and thus does not optimize for having the best reward.
- Greedy agents that opt to pull the arm with the highest action value, an agent determined expected value from an arm based on the average reward it has gotten thus far, do optimize for what they believe is the best reward. However, the agent can get stuck on suboptimal solutions due to noise in the distributions causing the optimal one to seem suboptimal.
- Thus, an agent needs a combination of being greedy whilst still allowing for exploration. Amongst the simplest examples is an epsilon greedy approach.

However, epsilon greedy using a fixed epsilon can be suboptimal, especially if the goal is stationary. Because of this many variants exist, for example, a time decaying epsilon greedy. When working with non-stationary targets, we want to alter the agent's expected reward for an arm so that more recent samples have a higher impact. One such example is using a recency-weighted average for updating the average reward of an arm. However, if noise is high and the weight for recent examples is also high it can also cause varying performance.

Amongst other things, the initial value of the expected reward can also impact the performance of an agent. Setting it to a large value (optimistic initial value) rather than zero for example can turn out to be better performing.

Other interesting approaches include SoftMax, however, as no agent knowledge should be used, determining a temperature is rather hard and thus this approach might not be ideal for this problem. Gradient bandits aim to solve this issue. Other interesting approaches include working with a regret where the received reward is compared with a theoretical best. But again, this would require domain knowledge. It is noted that this regret is specified in the provided code and that an optimal reward can be retrieved from the provided bandit class.

Whilst not being very useful in the general scheme of RL, Upper Confidence Bound (UCB) based action selection is known to be very effective for the bandit problem. It was discussed during the lecture that if you want to solve bandit problems, using UCB is a good approach. Since we are not allowed to use any extra information on the bandit except that we have several arms which give rewards, we opt for UCB as it was discussed this was a generally well-performing approach.


<hr>


## Discussion on the solution

As discussed, our solution is based on an Upper Confidence Bound (UCB) approach. Contrary to epsilon-greedy, UCB takes into account how it got the estimated reward for an arm to determine which action to perform next. For example, if we have two arms where one has an estimated reward of 2.8 and the other of 3.1, epsilon greedy would pick 3.1 in the greedy occasion. If both of these arms were pulled hundreds of times, this would be an excellent choice. However, if the arm of 3.1 was pulled a thousand times already and the arm of 2.8 was only pulled a few times, one might reasonably think it would be better to favour choosing the lower arm to see if the estimate is an accurate one. UCB builds on this idea to not rely solely on random exploration in epsilon greedy for this but work with a non-greedy action selection based on a type of potential given below: 
$$
A_t = argmax_a (Q_t(a) + c \sqrt{\frac{ln(t)}{N_t(a)}})
\\ \\
Q_t = \text{Action value estimate, favours exploitation}
\\ \\
c \sqrt{\frac{ln(t)}{N_t(a)}} = \text{Part for clever exploration based on Hoeffding's inequality}
\\ \\
c = \text{Weighting factor for favouring exploration}
\\ \\
ln(t) = \text{natural log of the current time step}
\\ \\
N_t = \text{number of times action a was performed so far}
$$


Thus, we don't choose completely random in the non-greedy case nor do we select only based on the amounts of times an arm is pulled but rather in a more human like manner where past experience is taken into account but weighted on the fact that old or small experience is less valuable.

It is noted that [the excellent explanatory video by ritvikmath](https://youtu.be/FgmMK6RPU1c) was used to gain further insight on UCB. It thought us that one area where UCB is known to struggle is when there are a lot of arms and a small number of timesteps. Having 32 arms and taking into account that an arm pull would take around an hour on a super computer, it is not unreasonable to think this could form a problem. However, we still believe as a general approach UCB is better than a more greedy approach which might have better results on limited timesteps but is expected to become worse again once timesteps increases.


<hr>


## Running the code

An Anaconda environment based on Python 3.8.10 was used for this homework. More information on this environment can be found in the [Anaconda environment documentation for the homework](../../documentation/README.md).

With the Anaconda Python environment installed as specified above, running the code is as simple as calling the `main.py` file and providing an optional argument that specifies the number of timesteps to take (defaults to 1000). An example call of the `main.py` file is as follows:

```bash
# Call the main.py file with an optional parameter specifying the number of timesteps to take
## Example of filled in call:
## & C:/Users/Lennert/.conda/envs/rl-homework/python.exe c:/fast_files/GitHub/VUB-RL/homework/1-bandits/main.py 2
& & {path/to/conda}/.conda/envs/rl-homework/python.exe {pat/to/github}/VUB-RL/homework/1-bandits/main.py 1000
```



