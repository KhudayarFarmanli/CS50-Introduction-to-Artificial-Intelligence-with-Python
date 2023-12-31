"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]



# Returns player who has the next turn on a board
def player(board):
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return X if x_count <= o_count else O
    

# Returns set of all possible actions (i, j) available on the board.
def actions(board):
    possible_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions
    

# Returns the board that results from making move (i, j) on the board.
def result(board, action):
    i, j = action
    if board[i][j] != EMPTY:
        raise Exception("Invalid action")
    new_board = [row[:] for row in board]  # Make a deep copy
    new_board[i][j] = player(board)
    return new_board
    

# Returns the winner of the game, if there is one.
def winner(board):
    for row in board:
        if row.count(X) == 3:
            return X
        elif row.count(O) == 3:
            return O

    for j in range(3):
        column = [board[i][j] for i in range(3)]
        if column.count(X) == 3:
            return X
        elif column.count(O) == 3:
            return O

    diagonals = [(board[0][0], board[1][1], board[2][2]), (board[0][2], board[1][1], board[2][0])]
    for diagonal in diagonals:
        if diagonal.count(X) == 3:
            return X
        elif diagonal.count(O) == 3:
            return O

    return None
    
    
# Returns True if game is over, False otherwise.
def terminal(board):
    return winner(board) is not None or all(all(cell != EMPTY for cell in row) for row in board)



# Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    
def utility(board):
    winner_player = winner(board)
    if winner_player == X:
        return 1
    elif winner_player == O:
        return -1
    return 0
    
# Computes the maximum value and corresponding action for the maximizing player (X)    
def max_value(board):
    if terminal(board):
        return utility(board), None

    value = float("-inf")
    best_action = None

    for action in actions(board):
        min_val, _ = min_value(result(board, action))
        if min_val > value:
            value = min_val
            best_action = action

    return value, best_action

# Computes the minimum value and corresponding action for the minimizing player (O)
def min_value(board):
    if terminal(board):
        return utility(board), None

    value = float("inf")
    best_action = None

    for action in actions(board):
        max_val, _ = max_value(result(board, action))
        if max_val < value:
            value = max_val
            best_action = action

    return value, best_action


# Returns the optimal action for the current player on the board.
def minimax(board):
    if terminal(board):
        return None

    current_player = player(board)
    if current_player == X:
        value, action = max_value(board)
    else:
        value, action = min_value(board)
    return action

