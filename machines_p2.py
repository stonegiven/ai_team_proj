import numpy as np
import random
from itertools import product

import time
import copy
BOARD_ROWS = 4
BOARD_COLS = 4
EXPLORATION_WEIGHT = 10
ITERATION_COUNT = 1000

pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces


def new_board():
    return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)

def new_pieces():
    return [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]

def detect_child(childNodes, data):
    for child in childNodes:
        if (child.data == data):
            return child
    return None

def detect_location(origin_board, new_board):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if origin_board[row][col] != new_board[row][col]:
                return (row, col)
    print('detect_locaiton 실패:')
    return None

def detect_available_locs(board):
    return [(row, col) for row, col in product(range(4), range(4)) if board[row][col]==0]

def check_line(line):
    if 0 in line:
        return False  # Incomplete line
    characteristics = np.array([pieces[piece_idx - 1] for piece_idx in line])
    for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
        if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
            return True
    return False

def check_2x2_subgrid_win(board):
    for r in range(BOARD_ROWS - 1):
        for c in range(BOARD_COLS - 1):
            subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
            if 0 not in subgrid:  # All cells must be filled
                characteristics = [pieces[idx - 1] for idx in subgrid]
                for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
                    if len(set(char[i] for char in characteristics)) == 1:  # All share the same characteristic
                        return True
    return False

def check_win(board):
    # Check rows, columns, and diagonals
    for col in range(BOARD_COLS):
        if check_line([board[row][col] for row in range(BOARD_ROWS)]):
            return True
    
    for row in range(BOARD_ROWS):
        if check_line([board[row][col] for col in range(BOARD_COLS)]):
            return True
        
    if check_line([board[i][i] for i in range(BOARD_ROWS)]) or check_line([board[i][BOARD_ROWS - i - 1] for i in range(BOARD_ROWS)]):
        return True

    # Check 2x2 sub-grids
    if check_2x2_subgrid_win(board):
        return True
    
    return False

class select_Node():
    def __init__(self, parent, piece, able_pieces, minmax):
        self.parent = parent
        self.data = piece
        self.able_pieces = able_pieces
        self.childNodes = []
        self.minmax = minmax
        if piece in parent.winning_pieces:
            self.win = True
        else:
            self.win = False
        self.x = 0
        self.n = 0
        self.N = 0
    
    def expand(self):
        parent = self.parent
        parent_board = parent.board

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if (parent_board[row][col] == 0):
                    if (not detect_child(self.childNodes,(row,col))):
                        newBoard = copy.deepcopy(parent.board)
                        newBoard[row][col] = pieces.index(self.data) + 1
                        newAblePiece = copy.deepcopy(self.able_pieces)
                        newAblePiece.remove(self.data)
                        newNode = place_Node(self, newBoard, (row,col), newAblePiece, not self.minmax)
                        self.childNodes.append(newNode)
        
class place_Node():
    def __init__(self, parent, board, location, able_pieces, minmax):
        self.parent = parent
        self.board = board
        self.data = location
        self.able_pieces = able_pieces        
        self.childNodes = []
        self.minmax = minmax
        self.winning_pieces = place_Node.get_winning_pieces(board, able_pieces)
        if (parent != None):
            self.N = parent.n
        self.n = 0
        self.x = 0
    
    def expand(self):
        for piece in self.able_pieces:
            if (not detect_child(self.childNodes, piece)):
                newAblePiece = copy.deepcopy(self.able_pieces)#.remove(piece)
                newNode = select_Node(self, piece, newAblePiece, self.minmax)
                self.childNodes.append(newNode)

    def get_winning_pieces(board, able_pieces):
        winning_pieces = []
        for piece in able_pieces:
            tmp_board = copy.deepcopy(board)
            for row in range(BOARD_ROWS):
                for col in range(BOARD_COLS):
                    if tmp_board[row][col] == 0:
                        tmp_board[row][col] = pieces.index(piece) + 1
                        if check_win(tmp_board):
                            winning_pieces.append(piece)
        return winning_pieces

class P2():
    head = place_Node(None, new_board() , None, new_pieces(), True)
    current_location = None
    current_board = new_board()
    current_piece = None
    def __init__(self, board, available_pieces):  # 생성자
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces

    def select_piece(self):
        if (P2.head.parent != None):
            head_set([P2.current_location])
        selected_piece = monte_carlo()
        print("P2: selected...", selected_piece)
        P2.current_piece = selected_piece
        P2.current_board = copy.deepcopy(self.board)
        return selected_piece

    def place_piece(self, selected_piece):
        head_set([P2.current_piece, detect_location(P2.current_board, self.board), selected_piece])
        position = monte_carlo()
        print("P2: placed...", position)
        P2.current_location = position
        return position
    
def head_set(list):
    for data in list:
        P2.head.expand()
        P2.head = detect_child(P2.head.childNodes, data)
        
        if (not P2.head):
            print(data,"실패")

def monte_carlo():
    root = P2.head
    iterations = ITERATION_COUNT

    for _ in range(iterations):
        node = root
        node.expand()

        while node.childNodes:
            node = select_best_child(node, False)

        x = simulate_game(node)
        backpropagate(node, x)

    best_child = select_best_child(root, True)
    return best_child.data


def select_best_child(node, getting_child):
    if not node.childNodes:
        if getting_child:
            print('고를게 없음...')
            return None
        else:
            return node

    def ucb1_score(child):
        if child.n == 0:
            return float('inf')  # Prioritize unexplored nodes
        exploitation = child.x
        exploration = EXPLORATION_WEIGHT * np.sqrt(np.log(node.n) / child.n)
        return exploitation + exploration
    
    # if (getting_child):
    #     for child in node.childNodes:
    #         if isinstance(child, select_Node):
                # if (child.win):
                    # print('winning piece remove', child.data)
                    # node.childNodes.remove(child)

    return max(node.childNodes, key=ucb1_score)

def simulate_game(node):
    if isinstance(node, select_Node):
        return simulate_game(node.parent)
    
    temp_board = copy.deepcopy(node.board)
    temp_able_pieces = copy.deepcopy(node.able_pieces)

    while not check_win(temp_board) and temp_able_pieces:
        piece = random.choice(temp_able_pieces)
        temp_able_pieces.remove(piece)

        available_locs = detect_available_locs(temp_board)
        if not available_locs:
            break

        row, col = random.choice(available_locs)
        temp_board[row][col] = pieces.index(piece) + 1

    if check_win(temp_board):
        return 1  # Current player wins
    return 0  # Draw

def backpropagate(node, x):

    while node is not None:
        if (not node.minmax):
            x = -x
        node.x = (node.n * node.x + x) / (node.n + 1)
        node.n += 1
        node = node.parent



