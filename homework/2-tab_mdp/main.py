#!/usr/bin/python3
#####################################################
# ABOUT THIS FILE
#####################################################
# Main file for RL Assignment 2: MDPs.
# Author: Lennert Bontick.
# Student ID: 0568702.
# VUB mail: lennert.bontinck@vub.be.

#####################################################
# REMARKS
#####################################################
# Example call: path/to/main.py.
# Please read the README.md file for more information on my solution.

#####################################################
# IMPORTS
#####################################################

import random as rnd
import numpy as np

from ice import *

#####################################################
# GLOBAL VARIABLES
#####################################################
# One can define the global variables here 

EPISODES = 100000 # Teacher provided: 100000 | Student: 100000
EPSILON = 0.02 # Rate of selecting random action - Teache provided: 0.1 | Student: 0.02
GAMMA = 0.2 # Discount factor - Teacher provided: 0.9 | Student: 0.2
LEARNING_RATE = 0.1 # Alpha - Teacher provided: 0.1 | Student: 0.1

#####################################################
# TEACHER PROVIDED FUNCTIONS
#####################################################
# The following functions were provided by the teacher
## We don't use them but leave them here in case they
## are needed by the teacher.

def argmax(l):
    """
    Return the index of the maximum element of a list
    """
    return max(enumerate(l), key=lambda x:x[1])[0]


#####################################################
# GLOBAL FUNCTIONS
#####################################################

numerical_action_to_string = {
    0: "up",
    1: "Down",
    2: "Left",
    3: "Right" 
}

def argmax_random(list_of_numbers):
    """
    Returns the index of the maximum value in a list, returns a random selection on ties.
    """
    # Get max indexes
    max_indexes = np.argwhere(list_of_numbers == np.max(list_of_numbers))
    
    # Flatten list of max indexes
    max_indexes = max_indexes.flatten()
    
    # Choose random max index
    return rnd.choice(max_indexes)

#####################################################
# AGENT CLASS
#####################################################

class DoubleQLearningIceAgent(object):
    def __init__(self, epsilon = EPSILON, alpha = LEARNING_RATE, gamma= GAMMA, initial_value = 0):    
        """
        Creates an agent which uses double Q learning in the ice.py environment
        """
        
        # Store agent specific information
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        # Create an instance of the environment to be used
        self.environment = Ice()
        
        # The agent can go up (0) down (1) left (2) right (3) in each state of the 4x4 grid
        self.possible_actions =  [[0, 1, 2, 3] for state in range(4*4)]
        
        # Store both Q tables
        self.Q1 = [[initial_value, initial_value, initial_value, initial_value] for state in range(4*4)]
        self.Q2 = [[initial_value, initial_value, initial_value, initial_value] for state in range(4*4)]
        
        # Set the initial state and game stats
        self.current_state = self.environment.reset()
        self.game_count = 1
        self.iteration_count = 0
        self.iteration_for_game_count = 0
        
        # Init other agent specific info
        self.total_reward = 0
        
        # Show progress
        print(f"Created an agent with: epsilon: {epsilon} | alpha: {alpha} | gamma: {gamma} | Initial values: {initial_value} ")
    
    def __random_action_for_state(self, state: int):    
        """
        Returns a random action for a given state.
        """
        return rnd.choice(self.possible_actions[state])
        
        
    def __max_action_for_state(self, state: int):    
        """
        Returns the action with the highest Q-value.
        Q-value is based on both Q1 as well as Q2.
        """
        Qcombined = [q1 + q2 for q1, q2 in zip(self.Q1[state], self.Q2[state])]
        
        return argmax_random(Qcombined)
    
    def reset_environment(self):  
        """
        Resets the environment and the agent's position in the environment (current_state).
        """
        # Reset environment
        self.current_state = self.environment.reset()
        
        # Playing new game, update agent's game stats
        self.game_count += 1
        self.iteration_for_game_count = 0
        
    def __update_q_values(self, current_state, action, reward, new_state):
        """
        Updates Q values of the agent for a given state, action and reward as well as the new state.
        """
        
        # Update Q1 or Q2 at random (i.e. each at randomly half the time)
        if (rnd.random() < 0.5):
            Q2value_for_Q1_max_action_in_new_state = self.Q2[new_state][argmax_random(self.Q1[new_state])]
            Q1value_for_current_state_action = self.Q1[current_state][action]
            
            self.Q1[current_state][action] += self.alpha * \
                (reward + \
                    (self.gamma * Q2value_for_Q1_max_action_in_new_state) -  Q1value_for_current_state_action)
        else:
            Q1value_for_Q2_max_action_in_new_state = self.Q1[new_state][argmax_random(self.Q2[new_state])]
            Q2value_for_current_state_action = self.Q2[current_state][action]
            
            self.Q2[current_state][action] += self.alpha * \
                (reward + \
                    (self.gamma * Q1value_for_Q2_max_action_in_new_state) -  Q2value_for_current_state_action)
            
    def __choose_action_epsilon_greedy(self):
        """
        Chooses an action for the current state in an epsilon greedy manner.
        """
        
        if (rnd.random() < self.epsilon):
            # Perform random action
            return self.__random_action_for_state(state= self.current_state)
        else:
            # Perform greedy action
            return self.__max_action_for_state(state= self.current_state)
        
        
    
    def __perform_move(self, action: int):  
        """
        Performs a given action on the current state and updates the agent's information accordingly.
        Returns wether or not the game was finished after this move.
        """
        
        # Perform the move
        new_state, obtained_reward, reached_termination = self.environment.step(action)
        
        # Update the game statistics
        self.iteration_count += 1
        self.iteration_for_game_count += 1
        self.total_reward += obtained_reward
        
        # Print move
        print(f"Iteration {self.iteration_count}: \t Game: \t {self.game_count} \t Old state: \t {self.current_state} \t Action: \t {numerical_action_to_string[action]} \t New state: {new_state} \t Obtained reward: \t {obtained_reward} \t Total reward: \t {self.total_reward}")
        
        # Update Q values
        self.__update_q_values(current_state= self.current_state,
                               action= action,
                               reward= obtained_reward,
                               new_state= new_state)
        
        # Move agent
        self.current_state = new_state
        
        # Check if end is reached
        if reached_termination:
            print(f"Reached end of a game after {self.iteration_for_game_count} steps in that game and got reward of {obtained_reward}.")
            print(f"Total of {self.game_count} games, average reward per game: {self.total_reward / self.game_count}, average reward per iteration: {self.total_reward / self.iteration_count}, total reward: {self.total_reward}")
            self.reset_environment()
            
            # End of game reached!
            return True
        else:
            # End of game not yet reached
            return False
        
    def perform_step(self):
        """
        Performs a singular step for the agent (i.e. execute an action and update the agent).
        Returns wether or not the game was finished after this move.
        """
        
        chosen_action = self.__choose_action_epsilon_greedy()
        
        game_ended = self.__perform_move(action= chosen_action)
        
        # Return whether or not game ended
        return game_ended
    

def main():
    
    # Create an agent
    agent = DoubleQLearningIceAgent(epsilon= EPSILON,
                                    alpha= LEARNING_RATE,
                                    gamma= GAMMA,
                                    initial_value= 0)
    
    # Loop over episodes
    for i in range(EPISODES):
        
        game_ended = False
        
        while(not game_ended):
            game_ended = agent.perform_step()

    # Print the final Q1 table
    print("\n\n Final Q1 table (max values per state shown): \n")
    for y in range(4):
        for x in range(4):
            print('%03.3f    \t' % max(agent.Q1[y*4 + x]), end='')
        print()
        
    print("\n\n Final Q1 table (best action per state): \n")
    for y in range(4):
        for x in range(4):
            print('%s\t\t' % numerical_action_to_string[argmax_random(agent.Q1[y*4 + x])], end='')
        print()
        

    # Print the final Q2 table
    print("\n\n Final Q2 table (max values per state shown): \n")
    for y in range(4):
        for x in range(4):
            print('%03.3f    \t' % max(agent.Q2[y*4 + x]), end='')
        print()
        
    print("\n\n Final Q2 table (best action per state): \n")
    for y in range(4):
        for x in range(4):
            print('%s\t\t' % numerical_action_to_string[argmax_random(agent.Q2[y*4 + x])], end='')
        print()
        

    # Print the combined Q table
    Qcombined = [[(q1 + q2)/2 for q1, q2 in zip(agent.Q1[state], agent.Q2[state])] for state in range(4*4)]
    print("\n\n Final combined Q table (max values per state shown): \n")
    for y in range(4):
        for x in range(4):
            print('%03.3f    \t' % max(Qcombined[y*4 + x]), end='')
        print()
        
    print("\n\n Final combined Q table (best action per state): \n")
    for y in range(4):
        for x in range(4):
            print('%s\t\t' % numerical_action_to_string[argmax_random(Qcombined[y*4 + x])], end='')
        print()

        

if __name__ == '__main__':
    main()
