# Reinforcement Learning Project @ VUB 2021 - 2022

This readme goes over the files and documents available for the RL project by Lennert Bontinck.

## Table of contents

- [Reinforcement Learning Project @ VUB 2021 - 2022](#reinforcement-learning-project--vub-2021---2022)
  - [Table of contents](#table-of-contents)
  - [Contact information](#contact-information)
  - [Setting up the correct Anaconda environment](#setting-up-the-correct-anaconda-environment)
  - [Experimental  notebooks](#experimental--notebooks)
  - [Paper specific notebooks](#paper-specific-notebooks)
  - [Base connect four pygame](#base-connect-four-pygame)
  - [Custom gym environment](#custom-gym-environment)
  - [Running the code](#running-the-code)

<hr>


## Contact information

| Name             | Student ID | VUB mail                                                  | Personal mail                                               |
| ---------------- | ---------- | --------------------------------------------------------- | ----------------------------------------------------------- |
| Lennert Bontinck | 0568702    | [lennert.bontinck@vub.be](mailto:lennert.bontinck@vub.be) | [info@lennertbontinck.com](mailto:info@lennertbontinck.com) |

<hr>


## Setting up the correct Anaconda environment

The RL project is based on a Python 3.8.10 Anaconda environment. To set up this environment, instructions and an Anaconda environment export is given as a `yml` file.

> More information on setting up the Anaconda environment can be found [here](../documentation/README.md).

<hr>


## Experimental  notebooks

During the development of the project some experimental notebooks were created to gain further insight on the Gym environment or performance of some algorithms on classical Gym environments.

| **Title**                               | **File location**                                            |
| --------------------------------------- | ------------------------------------------------------------ |
| 1: Testing Gym installation             | Available [here](experimental-notebooks/1-testing-gym-installation.ipynb) |
| 2: Testing custom Gym environment       | Available [here](experimental-notebooks/2-testing-custom-gym-environment.ipynb) |
| 3: Petting Zoo connect four environment | Available [here](experimental-notebooks/3-pettingzoo-connectfour.ipynb) |

<hr>


## Paper specific notebooks

For the paper of the project, multiple experiments were performed. The notebooks for these experiments are made available.

| **Title**                                                    | **File location**                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1: Connect four with random agents from Tianshou             | Available [here](paper_notebooks/1-learning-connect-four-random-agents-tianshou.ipynb) |
| 2: Connect four with Deep Q-Network vs Random agent from Tianshou | Available [here](paper_notebooks/2-learning-connect-four-dqn-vs-random-agent-tianshou.ipynb) |
| 3: Connect four with two Deep Q-Network agents from Tianshou | Available [here](paper_notebooks/3-learning-connect-four-dqn-agents-tianshou.ipynb) |
| 4: Improving DQN agents in connect four                      | Available [here](paper_notebooks/4-improving-dqn-agents.ipynb) |
| 5: Improving DQN architecture                                | Available [here](paper_notebooks/5-improving-dqn-architecture.ipynb) |



<hr>


## Base connect four pygame

A basic implementation of a connect four game was made. This game was made using pygame and is adopted from [Nihar99](https://github.com/Nihar99/pygame) and [Solomonleo12345](https://github.com/solomonleo12345/ConnectFour-Game). This game can be played in dual player mode by executing the following commands. By default this will open the game graphically in a popup window and print the board to the terminal.


```bash
# Activate the conda environment
conda activate rl-project

# Go to the GitHub folder of this project
cd path/to/GitHub/VUB-RL/project/base_connect4_pygame/

# Start a Jupyter notebook server
python connect_four.py
```



<hr>


## Custom gym environment

For the creation of the custom Gym environment, the [Gym documentation](https://www.gymlibrary.ml/content/environment_creation/) was used together with the supplied [Gym Examples GitHub repo](https://github.com/Farama-Foundation/gym-examples). To gain better insight on our connect four specific environment, comparable projects were studied such as those by [Andrei Suiu et al](https://github.com/IASIAI/gym-connect-four), [David Cotton](https://github.com/davidcotton/gym-connect4) and [Daniel Hernandez](https://github.com/Danielhp95/gym-connect4).

A v2 of the gym environment is also made, this differs from the original gym implementation in the way that it tries to mirror a Petting Zoo environment. This was important since the original environments of Gym don't have multi-agent settings. Petting Zoo does provide multi-agent settings in a semi-standardized manner. Due to it's popularity, this means most libraries supporting multi-agent gym environments rely on the Petting Zoo style of implementation.



<hr>


## Running the code

An Anaconda environment based on Python 3.8.10 was used for this homework. More information on this environment can be found in the [Anaconda environment documentation for the project](../../documentation/README.md).

With the Anaconda Python environment installed as specified above, the notebooks (`.ipynb` files) can be opened by running a Jupyter notebook server and navigating to them:

```bash
# Activate the conda environment
conda activate rl-project

# Go to the GitHub folder of this project
cd path/to/GitHub/VUB-RL/project/

# Start a Jupyter notebook server
jupyter notebook
```

