#!/usr/bin/python3
import sys
import numpy as np

import bandit

def main():
    timesteps = int(sys.argv[1])
    b = bandit.Bandit()
    
    regret = 0.

    for t in range(timesteps):
        # Choose an arm
        a = 0

        # Pull the arm, obtain a reward
        ret = b.trigger(a)
        regret += b.opt() - ret
        
        # Learn from a and ret
        print('Reward', ret, 'regret', regret)
        continue

if __name__ == '__main__':
    main()
