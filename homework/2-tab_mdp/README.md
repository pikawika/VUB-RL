# Assignment 2: Tabular Reinforcement Learning in an MDP

This readme file gives more information on the second assignment of the Reinforcement Learning (RL) course. It also includes the solution of Lennert Bontinck.

## Table of contents


- [Assignment 2: Tabular Reinforcement Learning in an MDP](#assignment-2-tabular-reinforcement-learning-in-an-mdp)
  - [Table of contents](#table-of-contents)
  - [Contact information](#contact-information)
  - [Assignment](#assignment)
  - [Challenge of the homework](#challenge-of-the-homework)
    - [About MDPs](#about-mdps)
    - [Solving the Bellman optimality equation](#solving-the-bellman-optimality-equation)
    - [Off-policy vs on-policy approaches](#off-policy-vs-on-policy-approaches)
  - [Discussion on the solution](#discussion-on-the-solution)
    - [Choosing an approach](#choosing-an-approach)
    - [Double Q-learning](#double-q-learning)
    - [Possible game paths and expected results](#possible-game-paths-and-expected-results)
    - [Results](#results)
    - [Further thoughts](#further-thoughts)
  - [Running the code](#running-the-code)


<hr>


## Contact information

| Name             | Student ID | VUB mail                                                  | Personal mail                                               |
| ---------------- | ---------- | --------------------------------------------------------- | ----------------------------------------------------------- |
| Lennert Bontinck | 0568702    | [lennert.bontinck@vub.be](mailto:lennert.bontinck@vub.be) | [info@lennertbontinck.com](mailto:info@lennertbontinck.com) |

<hr>


## Assignment

This assignment provides an environment (ice.py, an MDP), and the skeleton of a Reinforcement Learning agent. The agent always performs action 0, in any state. It provides a Q-Table, but does not learn anything nor updates any Q-Value. The goal of this project is to complete this agent with:

- A Q-Learning rule that allows correct Q-Values to be learned.
   - Beware that the environment is stochastic (need for a learning rate) and loopy (need for a discount factor if you don't want your agent to get stuck).
- An exploration strategy, for instance epsilon-greedy, that allows the agent to perform well in the environment and learn it fast.

It is possible to learn the task in one single (quite long) episode. You should therefore, at each time-step, print the current state, the action and the reward you obtain. This will allow us to measure the learning progress of your agent timestep per timestep.

The goal of this project is to learn the optimal policy as fast as possible. Bonus points are given for interesting exploration strategies, and for the project that achieves the fastest learning (ties both get the bonus).


<hr>


## Challenge of the homework

### About MDPs

The provided `ice.py` environment is a Markov Decision Process (MDP). MDPs are a mathematically idealized form of the reinforcement learning problem with specific properties. MDPs are a critical component of RL as discussed during the lectures. MDPs allow for formal definitions of RL problems and making theoretical statements. A typical MDP can be described as follows:

![Math](../imgs/2/mdp.png)

From this definition, one can see that it does allow for defining typical RL problems. Many variants of MDPs exist, with many of them often assuming certain properties such as the *Markov property*. Intuitively, this property means that the transition from the current state to the next state is only dependent on the current state not on the previous state the agent was in. Thus, all states of the MDP must include information about all aspects of the past agent-environment interaction that make a difference in the future. The handbook goes over these and other properties that are often assumed in more detail.

As typical RL problems can be represented as MDPs, trying to solve MDPs corresponds to solving the formalized RL problems. When we talk about solving an MDP, we talk about the goal of acting in the MDP such that the expected discounted return is maximized, given in the equation below.

![Math](../imgs/2/gt.png)

From this equation, it becomes visible how important it is to choose a good discount factor since a low lambda will result in a *myopic agent* (~greedy) and a high lambda will result in a *farsighted agent* (~exploratory). As a result, not only the reward but also the discount factor determine the goal. It is noted that for some situations the lambda can be 1, resulting in an undiscounted expected return.

Repeating all other definitions such as the *state-value function*, *action-value function* and *Bellman optimality equations* here is found to be unnecessary as this would be a simple repetition of the lecture and handbook. It is however important to note the difference between policy evaluation and control:

![Math](../imgs/2/policy.png)




### Solving the Bellman optimality equation

As discussed above, solving an MDP through acting in the MDP such that the expected discounted return is optimized corresponds to solving RL problems. Different approaches to doing this exist, during the lectures the following are discussed:

- Using a *perfect* model (i.e. dynamic programming)
   - Policy iteration
   - Value iteration
- Using sampling
   - Monte-Carlo (MC)
   - Q-learning
   - Sarsa

Dynamic programming, a solution relying on a perfect model, is of limited utility in RL since such a model is often not available or computationally very hard. This means approaches relying on such a perfect model won't be used for this assignment, although they are still very important, especially for theoretical aspects of RL.

Approaches using sampling are often called *model-free* approaches, although this term is a bit misleading since a model is still required. However, the model required is not a complete, perfect model but rather a model capable of generating experiences. Thus it does not have to provide complete probability distributions and differs greatly from the perfect model approaches, hence why it is sometimes called model-free. Model-free approaches are the backbone of RL, as providing a perfect model has been proven to be difficult even for the most simple games such as blackjack. Defining a win or a loss in blackjack is simple (model-free) but defining probabilities of winning at any given state can become hard very fast. 

Since this document does not aim to deliver a complete summary of the course, we won't discuss each approach in further detail, although our implemented approach will be discussed further in the `Discussion on the solution` section.

### Off-policy vs on-policy approaches

Having highlighted the difference between model-based and model-free approaches, another important differentiation between approaches is whether or not the approach is off-policy or on-policy. The difference between these two approaches is whether or not the policy being evaluated or improved is also used for generating data. On-policy evaluates or improves the policy used to make decisions. Off-policy evaluates or improves a policy different from what is used to generate data. Thus, off-policy requires two policies to be used: the *target* policy and the *behaviour* policy. The most simple example to intuitively show the difference is a situation as follows:

> We have an agent that uses an epsilon-greedy like approach, thus one that does not use the action with the highest expected reward all the time. However, to update estimated rewards, it uses the maximum possible reward for the next state, independent of the action chosen. This maximum is subsequently based on the best possible reward from its next state and so on. Thus, the expected rewards generated are based on an agent that acts greedy only. These two policies used differ, and thus we have an off-policy approach.

In general, on-policy is said to be simpler whilst off-policy requires more tinkering and results in greater variance and slower convergence. However, off-policy is often more general and can learn from data that is generated by a controller or a human expert. There is no single better option between on-policy or off-policy and it is dependent on the environment the agent acts in and what is wanted to be learned.

The risks and potential issues of using these approaches were further discussed in the lecture and handbook, with discussed concepts such as (weighted) importance sampling, maximization bias and other common issues that can arise from these strategies.


<hr>


## Discussion on the solution

After having analysed the problem in question and revising the theory, we decided which algorithm we found appropriate to implement. The reasoning behind this chosen algorithm, some properties and the results are discussed here.

### Choosing an approach

Q-learning is a very popular model-free off-policy reinforcement learning algorithm. It has proven to be very powerful whilst being relatively simple. However, as it uses an off-policy reward update rule where the maximum reward over all actions of the next state is used, there is the possibility of maximisation bias. This maximisation bias can result in poor policy decisions by the algorithm in certain stochastic environments, such as the class discussed cliff walking environment where SARSA resulted in a better overall policy as it opted for a safer route.

Whilst this would opt us to choose SARSA over Q-learning, we believe Q-learning is such a fundamental algorithm in RL it should not be dismissed and is probably more valuable from an educational aspect to implement. To make things more interesting, the variation on the classical Q-learning algorithm called double Q-learning is chosen. Double Q-learning aims to reduce the maximisation bias as discussed during the lecture. 

### Double Q-learning

As discussed, we opted to implement a double Q-learning algorithm for the agent to find an optimal policy in the provided `ice.py`environment. More information on the working of double Q-learning was gained from reading both the handbook and an [online article by Ziad Salloum](https://towardsdatascience.com/double-q-learning-the-easy-way-a924c4085ec3). The latter also had a sample python implementation of the double Q-learning algorithm which was used to gain further insight on how to implement it on our own.

Contrary to the first assignment, creating a true general agent is hard, as the agent relies on environment specific details such as the actions available per state etc. Whilst this could all be made to be passed as a parameter, this would make the code overly complex and thus it was opted to hardcode our agent class to this specific problem somewhat. It is noted that the double q learning algorithm implemented is based on the pseudocode for double q learning given in both the lecture as well as the handbook. Here, epsilon greedy makes use of both Q tables for selecting its next action.

### Possible game paths and expected results

When considering the environment the agent operates in, there are three possible paths an agent can *logically* take. Remember the game looks like this:

![Game](../imgs/2/grid.png)


Where:

![Game](../imgs/2/grid_states.png)

Since the environment is stochastic, it means that the action taken by the agent is not guaranteed to be executed. The first obvious path with a total reward of 100 looks like this:

![Game path](../imgs/2/path1.png)

An alternative path with a total reward of 120 is as follows:

![Game path](../imgs/2/path2.png)

Finally, such an agent might get stuck in a situation where it enters and leaves the *Treasure* state over and over again. If the environment were to be deterministic, this would result in a reward of 20 every other move. A possible approach to this could look like this:

![Game path](../imgs/2/path3.png)

The stochasticity of the environment is in the fact that a taken action might fail which results in the agent not moving. Thus, for the last possible path, this kind of stochasticity doesn't form an issue as the agent is not thrown into a pit resulting in a negative reward. The agent would only end up in a pit in that situation if the epsilon-greedy mechanism tries a random move which results in going into the pit.

This shows us that not only the reward and the discount factor may influence the agent but also the choosen epsilon value may influence the agent. Indeed, an agent with a high discount factor and low epsilon might converge to the last strategy, whilst one with a low discount factor but high epsilon might converge to the first discussed path.

### Results

TODO



### Further thoughts

TODO




<hr>


## Running the code

An Anaconda environment based on Python 3.8.10 was used for this homework. More information on this environment can be found in the [Anaconda environment documentation for the homework](../../documentation/README.md).

With the Anaconda Python environment installed as specified above, running the code is as simple as calling the `main.py` file.

In the top of the main.py file the most important parameters can be set.

```bash
# Call the main.py file with an optional parameter specifying the number of timesteps to take
## Example of filled in call:
## & C:/Users/Lennert/.conda/envs/rl-homework/python.exe c:/fast_files/GitHub/VUB-RL/homework/2-tab_mdp/main.py
& {path/to/conda}/.conda/envs/rl-homework/python.exe {path/to/github}/VUB-RL/homework/2-tab_mdp/main.py
```

