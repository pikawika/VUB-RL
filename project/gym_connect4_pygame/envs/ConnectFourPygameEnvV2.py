####################################################
# ABOUT THIS FILE
####################################################
# The base implementation of this connect four game comes from 2 open sources projects:
# Nihar99: https://github.com/Nihar99/pygame
# Solomonleo12345: https://github.com/solomonleo12345/ConnectFour-Game
#
# The edits and final used base game is provided on the GitHub of this project
#   under the base_connect4_pygame folder
# GitHub: https://github.com/pikawika/vub-rl
#
# This V2 is a modification of V1 to allow for using Tianshou multi-agent functionality.
# They follow a convetion which is similar to the coding style used by PettingZoo:
#   PettingZoo GitHub: https://github.com/Farama-Foundation/PettingZoo

####################################################
# INFO ABOUT THE AUTHOR
####################################################
# Name: Lennert Bontinck
# Email: lennert.bontinck@vub.be / info@lennertbontinck.com


####################################################
# IMPORTS
####################################################

# Gym for providing the environment
import gym

# Allow for optionals
from typing import Optional

# Pygame may be used for visualisation reasons
import pygame

# Numpy for easy numerical data structures
import numpy as np

# Allow for deep copies
import copy

# Use pettingzoo for generalised multi agent
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

####################################################
# GLOBAL VARIABLES
####################################################

# COLORS
COLOR_BLUE = (0, 0, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_YELLOW = (255, 255, 0)

# GRID CODES
GRID_EMPTY_SPACE = 0
GRID_PLAYER1_COIN = 1
GRID_PLAYER2_COIN = 2

# REWARDS
REWARD_WIN = 10
REWARD_LOSS = -10
REWARD_DRAW = 5
REWARD_INVALID = -1
REWARD_MOVE = 0
REWARD_BLOCKING = 0

####################################################
# GLOBAL FUNCTIONS FOR PETTING ZOO COMPATIBILITY
####################################################

def env(reward_win: int = REWARD_WIN,
        reward_loss: int = REWARD_LOSS,
        reward_draw: int = REWARD_DRAW,
        reward_invalid: int = REWARD_INVALID,
        reward_move: int = REWARD_MOVE,
        reward_blocking: int = REWARD_BLOCKING,
        allow_invalid_move: bool = True
        ):
    """
    Returns the environment with all it's wrappers.
    """
    
    env = raw_env(reward_win= reward_win,
                  reward_loss= reward_loss,
                  reward_draw= reward_draw,
                  reward_invalid= reward_invalid,
                  reward_move= reward_move,
                  reward_blocking= reward_blocking,
                  allow_invalid_move = allow_invalid_move)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=REWARD_INVALID)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def get_image(path):
    from os import path as os_path

    import pygame

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc

####################################################
# MAIN ENVIRONMENT CLASS
####################################################

class raw_env(AECEnv):
    """
    Main class for the Connect Four Gym environment which was adopted from a pygame.
    Supported render modes: "terminal", "human" | None defaults to terminal representation of board.
    """
    metadata = {
        "render_modes": ["terminal", "human"], # Supported render modes for visualisation
        "render_fps": 10,
        "is_parallelizable": False, # NOTE: edited for V2
        } 
    
    def __init__(self,
                 render_mode: Optional[str] = None,
                 grid_column_count: int = 7,
                 grid_row_count: int = 6,
                 reward_win: int = REWARD_WIN,
                 reward_loss: int = REWARD_LOSS,
                 reward_draw: int = REWARD_DRAW,
                 reward_invalid: int = REWARD_INVALID,
                 reward_move: int = REWARD_MOVE,
                 reward_blocking: int = REWARD_BLOCKING,
                 allow_invalid_move: bool = True):
        # Init from super which is a Petting Zoo class
        # NOTE: V2 edits w.r.t. PettingZoo and Tianshou multi-agent coding convention        
        super().__init__()
        
        # Ensure that a correct render mode is supplied
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Store game specific settings
        self.grid_column_count = grid_column_count 
        self.grid_row_count = grid_row_count
        self.reward_win = reward_win
        self.reward_loss = reward_loss
        self.reward_draw = reward_draw
        self.reward_invalid = reward_invalid
        self.reward_move = reward_move
        self.reward_blocking = reward_blocking
        self.allow_invalid_move = allow_invalid_move
        
        # Our game allows for two agents to play
        # NOTE: V2 edits w.r.t. PettingZoo and Tianshou multi-agent coding convention
        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]

        # The observation by an agent is the encoded form of a board with 3 possible int values (0= empty, 1=p1, 2=p2)
        # NOTE: V2 edits w.r.t. PettingZoo and Tianshou multi-agent coding convention
        self.observation_spaces = {
            i: gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(
                        low=0,
                        high=2,
                        shape=((self.grid_row_count, self.grid_column_count)),
                        dtype=np.int8
                        ),
                    "action_mask": gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.grid_column_count,),
                        dtype=np.int8),
                }
            )
            for i in self.agents
        }

        # The agents actions are in esence the possibility of placing a coin in each column
        # NOTE: V2 edits w.r.t. PettingZoo and Tianshou multi-agent coding convention
        self.action_spaces = {i: gym.spaces.Discrete(self.grid_column_count) for i in self.agents}

    def _get_obs(self):
        """
        Private function to get the observtions in the specified format.
        Our action mask always allows all columns as it should be learned that wrong pieces results in remaing the same agent.
        """
        # NOTE: V2 edits w.r.t. PettingZoo and Tianshou multi-agent coding convention
        if self.allow_invalid_move:
            return {
                "observation": self.__board,
                "action_mask": [1 for column in range(self.grid_column_count)]
                }
        else:
            return {
                "observation": self.__board,
                "action_mask": [self._is_valid_location(column) for column in range(self.grid_column_count)]
                }
            

    def _get_info(self):
        """
        Private function to get the info of a current state in the specified format.
        """
        return {
            "current_player": self.__current_players_coin
            }
        
    def reset(self, seed=None, options=None):
        """
        Resets the environment to an empty board.
        It is assumed reset is called before a first step call as per the Gym documentation.
        """
       
        # Create an initial empty board
        self.__board = self._empty_board()
        
        # Keep track of who's turn it is
        self.__player_one_playing = True
        self.__current_players_coin = GRID_PLAYER1_COIN if self.__player_one_playing else GRID_PLAYER2_COIN
    
        # Keep track if game is still playable
        self.__game_finished = False
        
        # Title is that player 1 should play
        self.__visual_title = "P1s TURN"
        
        # Clean the canvas in pygame
        if hasattr(self, '_raw_env__screen') and self.__screen is not None:
            self._draw_background_board_to_canvas()
            
            # Update screen with created canvas
            self.__screen.blit(self.__canvas, self.__canvas.get_rect())
            pygame.display.flip()
    
        # Reset Petting Zoo vars
        # NOTE: V2 updates
        self.agents = self.possible_agents[:]
        
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
         
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
    
    def step(self, action: int):
        """
        Performs an action (e.g. coin insert in provided column) on the current state of the board.
        Rewards are as follows:
            - Regular move: 0
            - Invalid move (e.g. full row): -1
            - Move leading to win: +10
            - Move leading to loss: -10
            - Move leading to draw: +5
        """

        # If the game was done, let it know using parent function
        # NOTE: edit for V2
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        
        # Update agent selection
        # NOTE: edit for v2 to be Petting Zoo like
        next_agent = self._agent_selector.next()
        
        # Keep a copy of the old board if we want to check for blocking reward
        if self.reward_blocking != 0:
            old_board = copy.deepcopy(self.__board)
        
        # Try to place the peace
        player_made_valid_move = self._place_piece_in_column(column= action)
        
        if not player_made_valid_move:
            # Title is that player made wrong move
            self.__visual_title = f"P{self.__current_players_coin} INVALID MOVE"
            
            # Board stays as it was, current agent gets negative reward, other no reward
            self.rewards[self.agent_selection] += self.reward_invalid
            self.rewards[next_agent] += 0
            self.dones = {i: False for i in self.agents}
            return
        
        # End game if player has won or give other oponent the turn
        elif self._winning_board():
            # Title is that player won
            self.__visual_title = f"!!! P{self.__current_players_coin} WON !!!"
            
            # Game is finished
            self.__game_finished = True
            
            # One player wins and the other loses
            self.rewards[self.agent_selection] += self.reward_win
            self.rewards[next_agent] += self.reward_loss
            self.dones = {i: True for i in self.agents}
            return
        
        # End game if full board without winners - draw
        elif self._full_board():
            # Title is that there is a tie
            self.__visual_title = "!!! TIE GAME !!!"
            
            # Game is finished
            self.__game_finished = True
            
            # Player made board filling move, tie
            self.rewards[self.agent_selection] += self.reward_draw
            self.rewards[next_agent] += self.reward_draw
            self.dones = {i: True for i in self.agents}
            return
        
        else:
            # Reward current agent for doing a move
            self.rewards[self.agent_selection] += self.reward_move
            self.rewards[next_agent] += 0
            
            # Check if a blocking move was made and a reward should be given
            if self.reward_blocking != 0:
                if self._blocking_move(board = old_board, action= action, oponent_coin= GRID_PLAYER2_COIN if self.__player_one_playing else GRID_PLAYER1_COIN):
                    # Reward current agent for doing a blocking move
                    self.rewards[self.agent_selection] += self.reward_blocking
                    self.rewards[next_agent] += 0
                
            
            # Game continues and switches to next player
            self.__player_one_playing = not self.__player_one_playing
            self.__current_players_coin = GRID_PLAYER1_COIN if self.__player_one_playing else GRID_PLAYER2_COIN
            
            # Update agent selection
            # NOTE: edit for v2 to be Petting Zoo like
            self.agent_selection = next_agent
            
            # Title is that it is the next player's turn
            self.__visual_title = f"P{self.__current_players_coin}s turn"
            
        # Accumelate the rewards
        self._accumulate_rewards()
    
    def render(self, mode='terminal'):
        """
        Renders the environment in the specified renderer mode.
        Defaults to "terminal" render.
        """
        if mode not in self.metadata["render_modes"]:
            raise NotImplementedError(f"test: Unknown render option, choose from: {self.metadata['render_modes']}")
        
        if mode == "terminal":
            # Print to the terminal
            print(np.flip(self.__board, 0))
            return
        
        if (not hasattr(self, '_raw_env__screen') and mode == "human") or (self.__screen is None and mode == "human"):
            # First time using human mode, init the pygame
            pygame.init()
            pygame.display.init()
            
            # Setup visual variables
            self.__visual_square_size = 100
            self.__visual_width= self.grid_column_count * self.__visual_square_size
            self.__visual_height = (self.grid_row_count + 1) * self.__visual_square_size # +1 for text banner
            self.__visual_screen_size = (self.__visual_width, self.__visual_height)
            self.__visual_coin_radius = int(self.__visual_square_size / 2 - 5)
            self.__visual_font = pygame.font.SysFont("monospace", 75)
            
            # Setup clock
            self.clock = pygame.time.Clock()
            
            # Setup the screen
            self.__screen = pygame.display.set_mode(self.__visual_screen_size)
            
            # Configure the canvas for hte screen
            self.__canvas = pygame.Surface(self.__visual_screen_size)
            
            # Render the background
            self._draw_background_board_to_canvas()
        
        if mode == "human":
            # Asumes background already drawn, update by showing coins
            self._draw_move_to_canvas()
            
            # Draws title to canvas
            self._draw_title_to_canvas()
            
            # Update screen with created canvas
            self.__screen.blit(self.__canvas, self.__canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            
            # Take into account FPS
            self.clock.tick(self.metadata["render_fps"])
            
            
        
        
    
    def close(self):
        """
        Closes the environment to free resources
        """
        if hasattr(self, '_raw_env__screen') and self.__screen is not None:
            # human mode was used, clear the pygame env
            pygame.display.quit()
            pygame.quit()
            self.__screen = None
            self.clock = None
        
        
        
        
            
    ####################################################
    # V2 ADDITIONS
    ####################################################
    # Functions added for V2
    
    def observation_space(self, agent):
        """
        Returns the observation space of a specified agent.
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """
        Returns the action space of a specified agent.
        """
        return self.action_spaces[agent]
    
    def observe(self, agent):
        """
        Returns the observation space.
        """
        return self._get_obs()
    
    def _legal_moves(self):
        """
        Returns the legal moves, which are all columns
        """
        return [i for i in range(self.grid_column_count)]
        
            
    ####################################################
    # HELPER FUNCTIONS FOR GAME LOGIC
    ####################################################
    # Functions adopted from base pygame
    
    def _empty_board(self):
        """
        Returns an empty board in the form of a row x column numpy ndarray.
        """
        board = np.zeros((self.grid_row_count, self.grid_column_count))
        return board

    def _is_valid_location(self, column: int):
        """
        Check if a column is playable.
        """
        # Column is playable if top piece is not yet set
        return self.__board[self.grid_row_count - 1][column] == GRID_EMPTY_SPACE

    def _get_free_space_row(self, column: int):
        """
        Returns the next open row for a specified column or -1 if no open row was found.
        """    
        # Loop all rows until a row is found where there is an empty space
        for row in range(self.grid_row_count):
            if self.__board[row][column] == GRID_EMPTY_SPACE:
                # Open row found, return it
                return row
            
        # No open rows found, return -1
        return -1

    def _place_piece_in_column(self, column: int):
        """
        Places a piece of a player in the specified column.
        Returns true if move was valid, false if the move was not valid and thus nothing was done.
        """
        if self._is_valid_location(column= column):
            # Determine the row by looking for first free space in that column
            free_space_row = self._get_free_space_row(column= column)
            
            # Place the coin
            self.__board[free_space_row][column] = self.__current_players_coin
            return True
        else: 
            return False
        
    def _winning_board(self):
        """
        Returns whether or not the board is won by the playing player.
        Should be called after placing a piece.
        """
        # check all horizontal locations
        for c in range(self.grid_column_count - 3):
            for r in range(self.grid_row_count):
                if self.__board[r][c] == self.__current_players_coin and self.__board[r][c + 1] == self.__current_players_coin and self.__board[r][c + 2] == self.__current_players_coin and \
                    self.__board[r][c + 3] == self.__current_players_coin:
                        return True
                    
        # check vertical locations for win
        for c in range(self.grid_column_count):
            for r in range(self.grid_row_count - 3):
                if self.__board[r][c] == self.__current_players_coin and self.__board[r + 1][c] == self.__current_players_coin and self.__board[r + 2][c] == self.__current_players_coin and \
                    self.__board[r + 3][c] == self.__current_players_coin:
                        return True

        # check positively sloped diagonals
        for c in range(self.grid_column_count - 3):
            for r in range(self.grid_row_count - 3):
                if self.__board[r][c] == self.__current_players_coin and self.__board[r + 1][c + 1] == self.__current_players_coin and self.__board[r + 2][c + 2] == self.__current_players_coin and \
                    self.__board[r + 3][c + 3] == self.__current_players_coin:
                        return True

        # check negatively sloped diagonals
        for c in range(self.grid_column_count - 3):
            for r in range(3, self.grid_row_count):
                if self.__board[r][c] == self.__current_players_coin and self.__board[r - 1][c + 1] == self.__current_players_coin and self.__board[r - 2][c + 2] == self.__current_players_coin and \
                    self.__board[r - 3][c + 3] == self.__current_players_coin:
                        return True
                
        # No winning board
        return False
        
    def _blocking_move(self, board, action, oponent_coin):
        """
        Returns whether or a given action by a given agent would be a blocking move.
        Needs the oponent coin to check this.
        """
        
        if board[self.grid_row_count - 1][action] != GRID_EMPTY_SPACE:
            # Invalid move and thus no blocking move
            return False
        
        # Place piece af is it were oponent piece
        for row in range(self.grid_row_count):
            if board[row][action] == GRID_EMPTY_SPACE:
                # Open row found, place piece and break loop
                board[row][action] = oponent_coin
                break
        
        # Check for win as before, if there is a win for the oponents piece, then the placed piece was a blocking piece.
        
        # check all horizontal locations
        for c in range(self.grid_column_count - 3):
            for r in range(self.grid_row_count):
                if board[r][c] == oponent_coin and board[r][c + 1] == oponent_coin and board[r][c + 2] == oponent_coin and board[r][c + 3] == oponent_coin:
                        return True
                    
        # check vertical locations for win
        for c in range(self.grid_column_count):
            for r in range(self.grid_row_count - 3):
                if board[r][c] == oponent_coin and board[r + 1][c] == oponent_coin and board[r + 2][c] == oponent_coin and board[r + 3][c] == oponent_coin:
                        return True

        # check positively sloped diagonals
        for c in range(self.grid_column_count - 3):
            for r in range(self.grid_row_count - 3):
                if board[r][c] == oponent_coin and board[r + 1][c + 1] == oponent_coin and board[r + 2][c + 2] == oponent_coin and board[r + 3][c + 3] == oponent_coin:
                        return True

        # check negatively sloped diagonals
        for c in range(self.grid_column_count - 3):
            for r in range(3, self.grid_row_count):
                if board[r][c] == oponent_coin and board[r - 1][c + 1] == oponent_coin and board[r - 2][c + 2] == oponent_coin and board[r - 3][c + 3] == oponent_coin:
                        return True
                
        # No winning board for oponent piece, so it was not a blocking piece
        return False
        
    def _full_board(self):
        """
        Checks if the board is full.
        Call this after winning_board to check for tie.
        """
        # If valid moves -> not full
        for column in range(self.grid_column_count):
            if self._is_valid_location(column):
                return False
            
        # No valid locations found, board full
        return True
        
        
        
        
            
    ####################################################
    # HELPER FUNCTIONS FOR VISUALISATION
    ####################################################
    # Functions adopted from base pygame
    
    def _draw_background_board_to_canvas(self):
        """
        Draws the board to the screen/pygame window.
        """
        # Check if pygame screen is available
        if self.__screen == None or self.__canvas == None:
            raise Exception("Working with pygame screen and canvas but None is configured.")
        
        # Fill with black (background color)
        self.__canvas.fill(COLOR_BLACK)
        
        # Draw the background
        for column in range(self.grid_column_count):
            for row in range(self.grid_row_count):
                # Draw a blue rectangle, e.g. the board
                pygame.draw.rect(self.__canvas, COLOR_BLUE, (column * self.__visual_square_size,
                                                             row * self.__visual_square_size + self.__visual_square_size, self.__visual_square_size, self.__visual_square_size))

                # Draw empty space if that space is empty
                pygame.draw.circle(self.__canvas, COLOR_BLACK, (int(column * self.__visual_square_size + self.__visual_square_size / 2),
                                                                int(row * self.__visual_square_size + self.__visual_square_size + self.__visual_square_size / 2)),
                                   self.__visual_coin_radius)
        
        # create the black rectangle on the top of the screen
        pygame.draw.rect(self.__canvas, COLOR_BLACK, (0, 0, self.__visual_width, self.__visual_square_size))
        
    def _draw_move_to_canvas(self):
        """
        Draws a new player move on the screen.
        Updats only what is needed to save time and memory.
        """     
        # Check if pygame screen is available
        if self.__screen == None or self.__canvas == None:
            raise Exception("Working with pygame screen and canvas but None is configured.")
        # Draw the player coins, needs to happen in seperate loop
        for column in range(self.grid_column_count):
            for row in range(self.grid_row_count):
                # Draw a red circle for player 1's coins
                
                if self.__board[row][column] == GRID_PLAYER1_COIN:
                    pygame.draw.circle(self.__canvas, COLOR_RED, (int(column * self.__visual_square_size + self.__visual_square_size / 2),
                                                                  self.__visual_height - int(row * self.__visual_square_size + self.__visual_square_size / 2)),
                                       self.__visual_coin_radius)
                    
                
                
                # Draw a yellow circle for player 2's coins
                elif self.__board[row][column] == GRID_PLAYER2_COIN:
                    pygame.draw.circle(self.__canvas, COLOR_YELLOW, (int(column * self.__visual_square_size + self.__visual_square_size / 2),
                                                                     self.__visual_height - int(row * self.__visual_square_size + self.__visual_square_size / 2)),
                                       self.__visual_coin_radius)
        
        
    def _draw_title_to_canvas(self):
        """
        Updates the title string to a given string.
        """
        # create the black rectangle on the top of the screen
        pygame.draw.rect(self.__canvas, COLOR_BLACK, (0, 0, self.__visual_width, self.__visual_square_size))
        
        # Determine the color of the text
        color = COLOR_RED if self.__player_one_playing else COLOR_YELLOW
            
        # render the font to a label
        label = self.__visual_font.render(self.__visual_title, True, color)

        # print the label on the screen
        self.__canvas.blit(label, (10, 10))
    



