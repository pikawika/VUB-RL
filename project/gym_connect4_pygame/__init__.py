####################################################
# INFO ABOUT THE AUTHOR
####################################################
# Name: Lennert Bontinck
# Email: lennert.bontinck@vub.be / info@lennertbontinck.com

from gym.envs.registration import register

register(
    id="lennert_bontinck/ConnectFour-v1", # ID to be used for environment creation
    entry_point="gym_connect4_pygame.envs:ConnectFourPygameEnvV1", # Path to main env file
    nondeterministic = False
)

register(
    id="lennert_bontinck/ConnectFour-v2", # ID to be used for environment creation
    entry_point="gym_connect4_pygame.envs:ConnectFourPygameEnvV2", # Path to main env file
    nondeterministic = False
)
