#!/usr/bin/python3
#####################################################
# ABOUT THIS FILE
#####################################################
# Main file for RL Assignment 1: Bandits.
# Author: Lennert Bontick.
# Student ID: 0568702.
# VUB mail: lennert.bontinck@vub.be.

#####################################################
# REMARKS
#####################################################
# Aproach should be universal without knowledge from the bandit (bandit.py).
# Example call: path/to/main.py 10, with 10 being the number of timesteps to take.
# Please read the README.md file for more information on my solution.
# The readme includes the overview of the experiment resulst also given in experiment_results.xlsx

#####################################################
# IMPORTS
#####################################################
import sys
import numpy as np
import random as rnd

import bandit as bd

#####################################################
# CLEARLY NAMED BANDIT FUNCTIONS
#####################################################
# Sort of wrapper functions for the bandit functions
## With clearer namings

def amount_of_arms_for_bandit(bandit: bd.Bandit):
    """
    Returns the amount of arms available for the provided bandit.
    """
    return bandit.num_arms()

def optimal_reward_for_bandit(bandit: bd.Bandit):
    """
    Returns the optimal reward available for the bandit.
    """
    return bandit.opt()

def pull_bandit_arm(bandit: bd.Bandit, arm: int):
    """
    Pulls the specified arm of the bandit and returns the received reward.
    """
    return bandit.trigger(arm)

#####################################################
# GENERAL FUNCTIONS
#####################################################
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

class Agent(object):
    def __init__(self, weighting_factor_c= 0.05, use_recency_weighted_average = False, recency_weight= 0.1, initial_value= None):    
        """
        Creates an agent that uses the specified weihgting factor c.
        """
        # Create a bandit object the agent has access to
        self.bandit = bd.Bandit()
        
        # Create a list of available arms
        # In this case it are indexes from 0 to the number of arms the bandits has (upper boundry not included)
        self.available_arms = [arm for arm in range(amount_of_arms_for_bandit(self.bandit))]
        
        # K-armed bandits is single state RL, there is only 1 state, we define it for completeness sake
        self.state = "singular_state"
        
        # Check if initial value should be optimal
        if (initial_value == None):
            initial_value = self.bandit.opt()
        
        # Create a list of the estimated reward for each arm
        self.estimated_rewards = [initial_value for arm in range(amount_of_arms_for_bandit(self.bandit))]
        
        # Keep track of overall reward
        self.overall_reward = 0
        
        # Create a list of the estimated reward for each arm
        self.arm_visit_counts = [0 for arm in range(amount_of_arms_for_bandit(self.bandit))]
        
        # Specify the strategy
        self.weighting_factor_c = weighting_factor_c
        self.recency_weight = recency_weight
        self.use_recency_weighted_average = use_recency_weighted_average
        
    def __estimated_reward_for_arm(self, arm: int):    
        """
        Gets the estimated reward for an arm so far.
        """
        return self.estimated_rewards[arm]
        
    def __uncertainty_for_arm(self, arm: int, timestep: int):    
        """
        Gets the uncertainty term for an arm on a given timestep.
        """
        if (self.arm_visit_counts[arm] < 1):
            # Hasn't been chosen yet, make uncertainty high
            return (self.weighting_factor_c) * 10
        else:
            return (self.weighting_factor_c) * (np.sqrt(np.log(timestep) / self.arm_visit_counts[arm]))
        
    def __ucb_for_arm(self, arm: int, timestep: int):    
        """
        Gets the UCB value for an arm at a given timestep
        """
        return self.__estimated_reward_for_arm(arm= arm) + self.__uncertainty_for_arm(arm= arm, timestep= timestep)
    
    def __ucb_for_all_arms(self, timestep: int): 
        """
        Gets the UCB value for all arms at a given timestep
        """
        return [self.__ucb_for_arm(arm= arm, timestep= timestep) for arm in self.available_arms]
    
    def __update_arm(self, arm: int, reward: float): 
        """
        Updates arm based on new reward gotten
        """
        # Update arm count
        self.arm_visit_counts[arm] += 1
        
        # Update estimated reward
        if self.use_recency_weighted_average:
            # based on recency-weighted average
            self.estimated_rewards[arm] = self.estimated_rewards[arm] + (self.recency_weight * (reward - self.estimated_rewards[arm]))
        else:
            # simple average method of updating estimated reward
            self.estimated_rewards[arm] = self.estimated_rewards[arm] + ((1/self.arm_visit_counts[arm]) * (reward - self.estimated_rewards[arm]))
        
        
    def perform_action(self, timestep: int):
        """
        Performs the agent selected action at the specified timestep
        Note: whilst we have access to the optimal reward allowing for regret based approaches,
                we don't use this info as the assignment asks to not use bandit specific info.
        """
        
        # Get all UCB values
        ucb_values = self.__ucb_for_all_arms(timestep= timestep)
        
        # Select arm with highest UCB, on ties select random
        arm_to_pull = argmax_random(ucb_values)
        
        # Pull the arm, obtain a reward
        reward = self.bandit.trigger(arm_to_pull)
        
        # Update arm
        self.__update_arm(arm= arm_to_pull, reward= reward)
        
        # Update overall reward
        self.overall_reward += reward
        
        # Print choice
        print(f"Timestep {timestep}: Pulled arm {arm_to_pull} for a reward of {reward}")
        
    def print_statistics(self):
        final_timestep = sum(self.arm_visit_counts)
        for arm in self.available_arms:
            print(f"Arm {arm}:\t Times chosen {self.arm_visit_counts[arm]} \t estimated reward: {round(self.estimated_rewards[arm], 5)} \t UCB for timestep {final_timestep}: {self.__ucb_for_arm(arm= arm, timestep= final_timestep)} ")
            
        print(f"\nOverall reward: {self.overall_reward}")
        print(f"Average reward per action: {self.overall_reward / sum(self.arm_visit_counts)}")
        print(f"Most chosen arm: {np.argmax(self.arm_visit_counts)} ({(np.max(self.arm_visit_counts) / final_timestep) * 100} %)")
        print(f"Optimal reward according to bandit: {self.bandit.opt()}")
        
        
        
        

#####################################################
# MAIN LOOP
#####################################################

def main():
    """
    Runs the game for the specified amount of timesteps (default: 120).
    Call: main.py [timesteps: int]
    Example call: main.py 120
    """
    
    # Check if argument is given that specifies the amount of timesteps to take
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        # Timesteps provided
        timesteps = int(sys.argv[1])
    else:
        # Default: to 10 timesteps
        timesteps = 120
        
    # Create bandit to know amount of arms, but delete bandit afterwards since it is created for the agent
    bandit = bd.Bandit()
    num_arms = bandit.num_arms()
    del bandit
    
    # Show info about bandit
    print("... INFORMATION ABOUT AGENT ....")
    
    # Depending on situation use more exploratory or more greedy approach
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
    
    # Show stats about bandit
    print(f"The Agents bandit has {len(agent.available_arms)} available arms.")
    
    # Do simulation
    print(f"\n\n... STARTING THE SIMULATION (specified {timesteps} timesteps) ...\n\n")
    
    for timestep in range(timesteps):
        agent.perform_action(timestep= timestep)
    
    
    print(f"\n\n... SIMULATION ENDED ...\n\n")
    
    # Show stats
    agent.print_statistics()

if __name__ == '__main__':
    main()
