####################################################
# ABOUT THIS FILE
####################################################
# The base implementation of this connect four game comes from 2 open sources projects:
# Nihar99: https://github.com/Nihar99/pygame
# Solomonleo12345: https://github.com/solomonleo12345/ConnectFour-Game

####################################################
# EDITS BY LENNERT BONTINCK
####################################################
# The base implementations were further edited by
# Name: Lennert Bontinck
# Email: lennert.bontinck@vub.be / info@lennertbontinck.com


####################################################
# IMPORTS
####################################################

import string
import numpy as np
import pygame
import sys
import math

####################################################
# INITIALIZE PYGAME
####################################################

# initialize pygame as some functions might need it straigh away
pygame.init()

####################################################
# GLOBAL VARIABLES
####################################################

# COLORS
COLOR_BLUE = (0, 0, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_YELLOW = (255, 255, 0)

# GRID SIZE
GRID_ROW_COUNT = 6
GRID_COLUMN_COUNT = 7

# GRID CODES
GRID_EMPTY_SPACE = 0
GRID_PLAYER1_COIN = 1
GRID_PLAYER2_COIN = 2

# VISUAL SETTINGS
VISUAL_SQUARESIZE = 100
VISUAL_WIDTH= GRID_COLUMN_COUNT * VISUAL_SQUARESIZE
VISUAL_HEIGHT = (GRID_ROW_COUNT + 1) * VISUAL_SQUARESIZE
VISUAL_SCREEN_SIZE = (VISUAL_WIDTH, VISUAL_HEIGHT)
VISUAL_COIN_RADIUS = int(VISUAL_SQUARESIZE / 2 - 5)
VISUAL_FONT = pygame.font.SysFont("monospace", 75)

# OTHER SETTINGS
LOGS_PRINT_TO_TERMINAL = True

####################################################
# GAME FUNCTIONS
####################################################

def empty_board():
    """
    Returns an empty board in the form of a row x column numpy ndarray.
    """
    board = np.zeros((GRID_ROW_COUNT, GRID_COLUMN_COUNT))
    return board

def is_valid_location(board: np.ndarray, column: int):
    """
    Check if a column is playable.
    """
    # Column is playable if top piece is not yet set
    return board[GRID_ROW_COUNT - 1][column] == GRID_EMPTY_SPACE

def get_free_space_row(board: np.ndarray, column: int):
    """
    Returns the next open row for a specified column or -1 if no open row was found.
    """    
    # Loop all rows until a row is found where there is an empty space
    for row in range(GRID_ROW_COUNT):
        if board[row][column] == GRID_EMPTY_SPACE:
            # Open row found, return it
            return row
        
    # No open rows found, return -1
    return -1


def place_piece_in_column(board: np.ndarray, column: int, grid_player_coin: int):
    """
    Places a piece of a player in the specified column.
    Returns true if move was valid, false if the move was not valid and thus nothing was done.
    """
    if is_valid_location(board, column):
        # Determine the row by looking for first free space in that column
        free_space_row = get_free_space_row(board= board,
                                            column= column)
        
        board[free_space_row][column] = grid_player_coin
        return True
    else: 
        return False


def print_board_to_terminal(board):
    """
    Prints the board to the terminal if logging set to that.
    """
    if LOGS_PRINT_TO_TERMINAL:
        print(np.flip(board, 0))
    

def winning_board(board: np.ndarray, grid_player_coin: int):
    """
    Returns whether or not the board is won by the provided player.
    Should be called after placing a piece.
    """
    # check all horizontal locations
    for c in range(GRID_COLUMN_COUNT - 3):
        for r in range(GRID_ROW_COUNT):
            if board[r][c] == grid_player_coin and board[r][c + 1] == grid_player_coin and board[r][c + 2] == grid_player_coin and \
                    board[r][c + 3] == grid_player_coin:
                return True

    # check vertical locations for win
    for c in range(GRID_COLUMN_COUNT):
        for r in range(GRID_ROW_COUNT - 3):
            if board[r][c] == grid_player_coin and board[r + 1][c] == grid_player_coin and board[r + 2][c] == grid_player_coin and \
                    board[r + 3][c] == grid_player_coin:
                return True

    # check positively sloped diagonals
    for c in range(GRID_COLUMN_COUNT - 3):
        for r in range(GRID_ROW_COUNT - 3):
            if board[r][c] == grid_player_coin and board[r + 1][c + 1] == grid_player_coin and board[r + 2][c + 2] == grid_player_coin and \
                    board[r + 3][c + 3] == grid_player_coin:
                return True

    # check negatively sloped diagonals
    for c in range(GRID_COLUMN_COUNT - 3):
        for r in range(3, GRID_ROW_COUNT):
            if board[r][c] == grid_player_coin and board[r - 1][c + 1] == grid_player_coin and board[r - 2][c + 2] == grid_player_coin and \
                    board[r - 3][c + 3] == grid_player_coin:
                return True
            
    # No winning board
    return False
    
def full_board(board: np.ndarray):
    """
    Checks if the board is full.
    Call this after winning_board to check for tie.
    """
    # If valid moves -> not full
    for column in range(GRID_COLUMN_COUNT):
        if is_valid_location(board, column):
            return False
        
    # No valid locations found, board full
    return True


def draw_background_board_to_display(screen):
    """
    Draws the board to the screen/pygame window.
    """
    screen.fill(COLOR_BLACK)
    
    # Draw the background
    for column in range(GRID_COLUMN_COUNT):
        for row in range(GRID_ROW_COUNT):
            # Draw a blue rectangle, e.g. the board
            pygame.draw.rect(screen, COLOR_BLUE, (column * VISUAL_SQUARESIZE,
                                                  row * VISUAL_SQUARESIZE + VISUAL_SQUARESIZE, VISUAL_SQUARESIZE, VISUAL_SQUARESIZE))

            # Draw empty space if that space is empty
            pygame.draw.circle(screen, COLOR_BLACK, (int(column * VISUAL_SQUARESIZE + VISUAL_SQUARESIZE / 2),
                                                        int(row * VISUAL_SQUARESIZE + VISUAL_SQUARESIZE + VISUAL_SQUARESIZE / 2)),
                                VISUAL_COIN_RADIUS)
    
    # create the black rectangle on the top of the screen
    pygame.draw.rect(screen, COLOR_BLACK, (0, 0, VISUAL_WIDTH, VISUAL_SQUARESIZE))
    
    # Update the screen    
    pygame.display.update()
    pygame.display.flip()
    
def draw_move_to_display(board: np.ndarray, screen):
    """
    Draws a new player move on the screen.
    Updats only what is needed to save time and memory.
    """     
    # Draw the player coins, needs to happen in seperate loop
    for column in range(GRID_COLUMN_COUNT):
        for row in range(GRID_ROW_COUNT):
            # Draw a red circle for player 1's coins
            
            if board[row][column] == GRID_PLAYER1_COIN:
                pygame.draw.circle(screen, COLOR_RED, (int(column * VISUAL_SQUARESIZE + VISUAL_SQUARESIZE / 2),
                                                       VISUAL_HEIGHT - int(row * VISUAL_SQUARESIZE + VISUAL_SQUARESIZE / 2)),
                                   VISUAL_COIN_RADIUS)
                
            
            
            # Draw a yellow circle for player 2's coins
            elif board[row][column] == GRID_PLAYER2_COIN:
                pygame.draw.circle(screen, COLOR_YELLOW, (int(column * VISUAL_SQUARESIZE + VISUAL_SQUARESIZE / 2),
                                                          VISUAL_HEIGHT - int(row * VISUAL_SQUARESIZE + VISUAL_SQUARESIZE / 2)),
                                   VISUAL_COIN_RADIUS)
    
    # Update the screen                
    pygame.display.update()

def update_board_display(board: np.ndarray, screen):
    """
    Updates both pygame's visual board and terminal printed board.
    """
    print_board_to_terminal(board= board)
    draw_move_to_display(board= board, screen= screen)
    
    
def update_title_text(screen, title: string, color):
    """
    Updates the title string to a given string.
    """
    # create the black rectangle on the top of the screen
    pygame.draw.rect(screen, COLOR_BLACK, (0, 0, VISUAL_WIDTH, VISUAL_SQUARESIZE))
    
    # render the font to a label
    label = VISUAL_FONT.render(title, True, color)

    # print the label on the screen
    screen.blit(label, (10, 10))
    
    # Update the screen             
    pygame.display.update()
    

# Main function responsible for playing the game
def main():
    # Create an initial empty board
    board = empty_board()

    # Keep track of who's turn it is
    player_one_playing = True
    
    # Keep track if game is still playable
    game_finished = False

    # set the pygame screen to our resolution
    screen = pygame.display.set_mode(VISUAL_SCREEN_SIZE)
    
    # Show the background
    draw_background_board_to_display(screen= screen)
    
    # Show initial message
    update_title_text(screen= screen,
                        title= f"P1: MAKE MOVE",
                        color= COLOR_RED)
    

    # the main loop of the game
    while not game_finished:
        # Keep track of events on pygame window
        for event in pygame.event.get():
            
            # User clicked cross/exit
            if event.type == pygame.QUIT:
                sys.exit()

            # Mouse click = user input
            if event.type == pygame.MOUSEBUTTONDOWN:
                
                # Determine column from the x position of the mouse click
                x_coordinates_mouse = event.pos[0]
                user_selected_column = int(math.floor(x_coordinates_mouse / VISUAL_SQUARESIZE))
                
                # Determine current player
                current_players_coin = GRID_PLAYER1_COIN if player_one_playing else GRID_PLAYER2_COIN
                current_players_color = COLOR_RED if player_one_playing else COLOR_YELLOW

                # Try to place the peace
                player_made_valid_move = place_piece_in_column(board= board, 
                                                               column= user_selected_column,
                                                               grid_player_coin= current_players_coin)
                
                if not player_made_valid_move:
                    # Player made invalid move, show warning
                    update_title_text(screen= screen,
                                      title= f"P{current_players_coin}: RETRY (FULL)",
                                      color= current_players_color)
                    
                    # No further execution since invalid move:
                    continue
                
                # Draw update
                update_board_display(board, screen)

                # End game if player has won or give other oponent the turn
                if winning_board(board= board, 
                                 grid_player_coin= current_players_coin):

                    # Print winning message
                    update_title_text(screen= screen,
                                      title= f"!!! P{current_players_coin} WON !!!",
                                      color= current_players_color)
                    
                    # Game is finished
                    game_finished = True
                    
                    # wait for 4 seconds before closing the game
                    pygame.time.wait(4000)
                elif full_board(board):
                    # Print winning message
                    update_title_text(screen= screen,
                                      title= f"!!! TIE !!!",
                                      color= COLOR_BLUE)
                    
                    # Game is finished
                    game_finished = True
                    
                    # wait for 4 seconds before closing the game
                    pygame.time.wait(4000)
                else:
                    # Alternate the player's turn
                    player_one_playing = not player_one_playing
                    current_players_coin = GRID_PLAYER1_COIN if player_one_playing else GRID_PLAYER2_COIN
                    current_players_color = COLOR_RED if player_one_playing else COLOR_YELLOW
                    
                    # Show new message
                    update_title_text(screen= screen,
                                    title= f"P{current_players_coin}: MAKE MOVE",
                                    color= current_players_color)

                    

if __name__ == "__main__":
    main()


