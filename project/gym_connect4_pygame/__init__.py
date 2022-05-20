####################################################
# INFO ABOUT THE AUTHOR
####################################################
# Name: Lennert Bontinck
# Email: lennert.bontinck@vub.be / info@lennertbontinck.com

from gym.envs.registration import register

register(
    id="lennert_bontinck/ConnectFour-v1", # ID to be used for environment creation
    entry_point="gym_connect4_pygame.envs:ConnectFourPygameEnv", # Path to main env file
    nondeterministic = False
)
