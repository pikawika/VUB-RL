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

#####################################################
# IMPORTS
#####################################################
import sys
import numpy as np

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
# AGENT CLASS
#####################################################

class Agent(object):
    def __init__(self):
        # Create a bandit object
        self.bandit = bd.Bandit()
        
        # Create a list of available arms
        # In this case it are indexes from 0 to the number of arms the bandits has (upper boundry not included)
        self.available_arms = [arm for arm in range(amount_of_arms_for_bandit(self.bandit))]
        
        # K-armed bandits is single state RL, there is only 1 state, we define it for completeness sake
        self.state = "singular_state"

#####################################################
# MAIN LOOP
#####################################################

def main():
    """
    Runs the game for the specified amount of timesteps.
    Call: main.py [timesteps: int]
    Example call: main.py 10
    """
    
    # Check if argument is given that specifies the amount of timesteps to take
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        # Timesteps provided
        timesteps = int(sys.argv[1])
        print(f"\n\n... STARTING THE SIMULATION (default {timesteps} timesteps) ...\n\n")
    else:
        # Default: to 10 timesteps
        timesteps = 10
        print(f"\n\n... STARTING THE SIMULATION (specified {timesteps} timesteps) ...\n\n")
        
    # Make a bandit object
    bandit = bd.Bandit()
    
    # Show stats about bandit
    print("... INFORMATION ABOUT BANDIT ....")
    print(f"My bandit has {amount_of_arms_for_bandit(bandit)} arms.")
    print(f"The optimal award from all arms is {optimal_reward_for_bandit(bandit)}.")
    
    # TODO: implement strategy below, sample code by professor
    # Init variables
    regret = 0.

    for timestep in range(timesteps):
        # Choose an arm
        action = 1

        # Pull the arm, obtain a reward
        reward = bandit.trigger(action)
        
        # Calculate regret based on what the max mean reward from all arms is
        regret += bandit.opt() - reward
        
        # Learn from reward and regret
        # print('Reward', reward, 'regret', regret)
        continue

if __name__ == '__main__':
    main()
