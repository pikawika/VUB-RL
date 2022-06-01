
####################################################
# ABOUT THIS FILE
####################################################
# Description: MiniMax agent for connect four
#
# Author name: Lennert Bontinck
# Author email: lennert.bontinck@vub.be / info@lennertbontinck.com
#
# File based on: https://github.com/KeithGalli/Connect4-Python/blob/master/connect4_with_ai.py


####################################################
# IMPORTS
####################################################

import math
import random as rnd
import copy
import numpy as np

####################################################
# MINIMAX AGENT
####################################################

class MiniMaxConnectFourBot:
    def __init__(self, coin: int, oponent_coin: int, column_count: int, row_count: int, minimax_depth: int):
        """
        Creates a MiniMax bot for our custom connect four application.
        Based on: https://github.com/KeithGalli/Connect4-Python/blob/master/connect4_with_ai.py
        """     
        self.coin = coin
        self.oponent_coin = oponent_coin
        self.empty_space = 0
        self.column_count = column_count
        self.row_count = row_count
        self.minimax_depth = minimax_depth
        
    def predict(self, board):
        # Get chosen col based on minimax score
        chosen_col, minimax_score = self.__minimax(board= board,
                                                   depth= self.minimax_depth)
        
        # Return the chosen col
        return chosen_col
    
    def __get_valid_locations(self, board):
        """
        Gets valid locations.
        Taken from: https://github.com/KeithGalli/Connect4-Python/blob/master/connect4_with_ai.py
        """    
        valid_locations = []
        for col in range(self.column_count):
            if board[self.row_count - 1][col] == self.empty_space:
                valid_locations.append(col)
        return valid_locations
    
    def __get_next_open_row(self, board, col):
        # Loop all rows until a row is found where there is an empty space
        for row in range(self.row_count):
            if board[row][col] == self.empty_space:
                # Open row found, return it
                return row
            
        # No open rows found, return -1
        return -1
            
    def __evaluate_window(self, window, piece):
        opponent_piece = self.oponent_coin

        # initial score of a window is 0
        score = 0

        # based on how many friendly pieces there are in the window, we increase the score
        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 2

        # or decrese it if the oponent has 3 in a row
        if window.count(opponent_piece) == 3 and window.count(0) == 1:
            score -= 4 

        return score 
            
    def __score_position(self, board):

        score = 0
        piece = self.coin

        # score center column --> we are prioritizing the central column because it provides more potential winning windows
        center_array = [int(i) for i in list(board[:,self.column_count//2])]
        center_count = center_array.count(piece)
        score += center_count * 6

        # below we go over every single window in different directions and adding up their values to the score
        # score horizontal
        for r in range(self.row_count):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(self.column_count - 3):
                window = row_array[c:c + 4]
                score += self.__evaluate_window(window, piece)

        # score vertical
        for c in range(self.column_count):
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(self.row_count-3):
                window = col_array[r:r+4]
                score += self.__evaluate_window(window, piece)

        # score positively sloped diagonals
        for r in range(3,self.row_count):
            for c in range(self.column_count - 3):
                window = [board[r-i][c+i] for i in range(4)]
                score += self.__evaluate_window(window, piece)

        # score negatively sloped diagonals
        for r in range(3,self.row_count):
            for c in range(3,self.column_count):
                window = [board[r-i][c-i] for i in range(4)]
                score += self.__evaluate_window(window, piece)

        return score
    
    def __winning_board(self, board: np.ndarray, grid_player_coin: int):
        """
        Returns whether or not the board is won by the provided player.
        Should be called after placing a piece.
        """
        
        # check all horizontal locations
        for c in range(self.column_count - 3):
            for r in range(self.row_count):
                if board[r][c] == grid_player_coin and board[r][c + 1] == grid_player_coin and board[r][c + 2] == grid_player_coin and \
                        board[r][c + 3] == grid_player_coin:
                    return True

        # check vertical locations for win
        for c in range(self.column_count):
            for r in range(self.row_count - 3):
                if board[r][c] == grid_player_coin and board[r + 1][c] == grid_player_coin and board[r + 2][c] == grid_player_coin and \
                        board[r + 3][c] == grid_player_coin:
                    return True

        # check positively sloped diagonals
        for c in range(self.column_count - 3):
            for r in range(self.row_count - 3):
                if board[r][c] == grid_player_coin and board[r + 1][c + 1] == grid_player_coin and board[r + 2][c + 2] == grid_player_coin and \
                        board[r + 3][c + 3] == grid_player_coin:
                    return True

        # check negatively sloped diagonals
        for c in range(self.column_count - 3):
            for r in range(3, self.row_count):
                if board[r][c] == grid_player_coin and board[r - 1][c + 1] == grid_player_coin and board[r - 2][c + 2] == grid_player_coin and \
                        board[r - 3][c + 3] == grid_player_coin:
                    return True
                
        # No winning board
        return False
    
    def __minimax(self, board: np.array, depth, alpha=-math.inf, beta= math.inf, maximizing_player: bool = True):
        """
        Does minimax prediction for best column.
        Based on: https://github.com/KeithGalli/Connect4-Python/blob/master/connect4_with_ai.py
        """     
        # Get the valid locations
        valid_locations = self.__get_valid_locations(board)

        # Stop condition - reached winning board
        if self.__winning_board(board= board, grid_player_coin= self.coin):
            return (None, 100000000000000)

        # Stop condition - reached losing board
        if self.__winning_board(board= board, grid_player_coin= self.oponent_coin):
            return (None, -100000000000000)

        # Stop condition - full board
        if np.count_nonzero(board == self.empty_space) == 0:
            return (None, 0)

        # Stop condition - reached depth
        if depth == 0:
            return (None, self.__score_position(board)) 
            
        if (maximizing_player): 
            # Find best column based on max value
            value = -math.inf
            column = rnd.choice(valid_locations)  
                     
            # recursive minimax
            for col in valid_locations:
                
                # Place piece in copy of board
                row = self.__get_next_open_row(board, col)
                b_copy = copy.deepcopy(board)
                b_copy[row][col] = self.coin
                
                # Get score recursive
                new_score = self.__minimax(board= b_copy,
                                           depth= depth-1,
                                           alpha= alpha,
                                           beta= beta,
                                           maximizing_player= False)[1]
                
                # Update best 
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                
                # Alpha beta pruning
                if alpha >= beta:
                    break
                
            return column, value
        
        else:
            # Find best column based on min value
            value = math.inf
            column = rnd.choice(valid_locations)
            for col in valid_locations:
                # Place piece in copy of board
                row =  self.__get_next_open_row(board, col)
                b_copy = copy.deepcopy(board)
                b_copy[row][col] = self.oponent_coin
                
                # Get score recursive
                new_score = self.__minimax(board= b_copy,
                                           depth= depth-1,
                                           alpha= alpha,
                                           beta= beta,
                                           maximizing_player= True)[1]
                
                # Update best 
                if new_score < value:
                    value = new_score
                    column = col
                    
                # Alpha beta pruning
                beta = min(value, beta) 
                if alpha >= beta:
                    break
                
            return column, value
