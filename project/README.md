# Reinforcement Learning Project @ VUB 2021 - 2022

This readme goes over the files and documents available for the RL project by Lennert Bontinck.

## Table of contents

- [Reinforcement Learning Project @ VUB 2021 - 2022](#reinforcement-learning-project--vub-2021---2022)
  - [Table of contents](#table-of-contents)
  - [Contact information](#contact-information)
  - [Setting up the correct Anaconda environment](#setting-up-the-correct-anaconda-environment)
  - [Experimental  notebooks](#experimental--notebooks)
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

| **Title**                   | **File location**                                            |
| --------------------------- | ------------------------------------------------------------ |
| 1: Testing Gym installation | Available [here](experimental-notebooks/1-testing-gym-installation.ipynb) |



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

