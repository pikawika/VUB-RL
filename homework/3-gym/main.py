import random
import gym
import sys

EPISODES = 100000
EPSILON = 0.1
GAMMA = 0.9
LEARNING_RATE = 0.1

def argmax(l):
    """ Return the index of the maximum element of a list
    """
    return max(enumerate(l), key=lambda x:x[1])[0]

def main():
    # Create the Gym environment. A good first environment is FrozenLake-v0
    env = gym.make(sys.argv[1])
    print(env.action_space)
    print(env.observation_space)

    # Act randomly in the environment
    average_cumulative_reward = 0.0

    # Loop over episodes
    for i in range(EPISODES):
        state = env.reset()
        terminate = False
        cumulative_reward = 0.0

        # Loop over time-steps
        while not terminate:
            # Choose an action at random
            a = env.action_space.sample()       # Note: with discrete actions, env.action_space.n is the number of actions in the environment

            # Perform the action
            next_state, r, terminate, info = env.step(a)

            # Update statistics
            cumulative_reward += r
            state = next_state

        # Per-episode statistics
        average_cumulative_reward *= 0.95
        average_cumulative_reward += 0.05 * cumulative_reward

        print(i, cumulative_reward, average_cumulative_reward)

if __name__ == '__main__':
    main()
