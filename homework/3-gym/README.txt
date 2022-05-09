Reinforcement Learning in the OpenAI Gym
========================================

This assignment introduces you to the wonderful world of the OpenAI gym. With a very simple interface, the Gym allows you to interact with a plethora of interesting environments, of which:

- Simulated 3D robots (Mujoco and Roboschool)
- The Atari games
- The OpenAI Gym Retro package allows you agent to play any GameBox, NES, SNES, GameCube or Sega Genesis game

This assignment contains a mostly-empty main.py file, that shows how to create a Gym environment. Your tasks are the following:

- 5 points: Create a simple tabular Q-Learning agent (same as in Exercise 2), with the exploration strategy of your liking. The points given here come from the fact that you may have to get used to Python, coming from a Matlab background.
- 5 points: extend that agent with Experience Replay.
- 5 points: find a way to support continuous-states environment (instead of numbered states, your agent observes Numpy arrays that are vectors of floats, such as x,y coordinates in a room). You can use neural networks (seen on Friday, we recomment the PyTorch or Keras Python libraries, they are very easy to use), or any other creative way. Test this with a simple environment such as LunarLander.
- 5 points: successfully learn LunarLander. LunarLander is successfully learned when the mean cumulative reward reaches 200.
