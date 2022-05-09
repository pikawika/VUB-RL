# Assignment 3: Reinforcement Learning in the OpenAI Gym

This readme file gives more information on the third and final assignment of the Reinforcement Learning (RL) course. It also includes the solution of Lennert Bontinck.

## Table of contents

- [Contact information](#contact-information)
- [Assignment](#assignment)
   - [About OpenAI Gym](#about-openai-gym)
   - [tasks](#tasks)

- [TODO](#todo)

<hr>


## Contact information

| Name             | Student ID | VUB mail                                                  | Personal mail                                               |
| ---------------- | ---------- | --------------------------------------------------------- | ----------------------------------------------------------- |
| Lennert Bontinck | 0568702    | [lennert.bontinck@vub.be](mailto:lennert.bontinck@vub.be) | [info@lennertbontinck.com](mailto:info@lennertbontinck.com) |

<hr>


## Assignment

### About OpenAI Gym

This assignment introduces you to the wonderful world of the OpenAI gym. With a very simple interface, the Gym allows you to interact with a plethora of interesting environments, of which:

- Simulated 3D robots (Mujoco and Roboschool)
- The Atari games
- The OpenAI Gym Retro package allows you agent to play any GameBox, NES, SNES, GameCube or Sega Genesis game

### Tasks

This assignment contains a mostly-empty main.py file, that shows how to create a Gym environment. Your tasks are the following:

- Create a simple tabular Q-Learning agent (same as in Exercise 2), with the exploration strategy of your liking.
   - 5 points
   - The points given here come from the fact that you may have to get used to Python, coming from a Matlab background.
- Extend that agent with Experience Replay.
   - 5 points
- Find a way to support continuous-states environment (instead of numbered states, your agent observes Numpy arrays that are vectors of floats, such as x,y coordinates in a room). You can use neural networks (seen on Friday, we recomment the PyTorch or Keras Python libraries, they are very easy to use), or any other creative way. Test this with a simple environment such as LunarLander.
   - 5 points
- Successfully learn LunarLander. LunarLander is successfully learned when the mean cumulative reward reaches 200.
   - 5 points


<hr>


## TODO

TODO
