import numpy as np
import random
from itertools import product

import time
import copy
BOARD_ROWS = 4
BOARD_COLS = 4
EXPLORATION_WEIGHT = 10
WEIGHT_CHANGE = 0.03
# CALC_TIME = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# CALC_TIME = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
CALC_TIME = [1, 3, 6, 8, 10, 12, 15, 18, 20, 22, 25, 27, 29, 32, 35, 37, 37, 35, 32, 29, 27, 25, 22, 20, 18, 15, 12, 10, 8, 6, 3, 1]


pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces

# 2진수의 piece를 mbti문자열로 반환
def toString_mbti(tuple):
    str = ''
    if (tuple[0] == 0):
        str += 'I'
    else:
        str += 'E'
    if (tuple[1] == 0):
        str += 'N'
    else:
        str += 'S'
    if (tuple[2] == 0):
        str += 'T'
    else:
        str += 'F'
    if (tuple[3] == 0):
        str += 'P'
    else:
        str += 'J'
    return str

# monte carlo의 모든 과정을 print (디버깅용)
def print_monte_carlo(node, depth):
    print('depth..............................',depth)
    node.print_it()
    print('x : ',node.x,'n : ', node.n)
    for child in node.childNodes:
        if child.n != 0:
            print_monte_carlo(child, depth + 1)

# 보드출력 (디버깅용)
def print_board(board):
    for row in board:
        list = []
        for col in row:
            if (col == 0):
                list.append('    ')
            else:
                list.append(toString_mbti(pieces[col-1]))
        print(list)

# 데이터 출력 (디버깅용)
def print_child_data(node):
    for child in node.childNodes:
        print("..........................",child.depth)
        print_board(child.board)
        print('x:',child.x)
        print('n: ',child.n)
        print('clear:', child.clear)
        print(child.data)

# 빈 보드 하나 생성
def new_board():
    return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)

# 모든 piece가 들은 배열 생성
def new_pieces():
    return [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]

# childNodes배열의 child중, 특정 data(selected_piece or place_location)를 가진놈을 반환
def detect_child(childNodes, data):
    for child in childNodes:
        if (child.data == data):
            return child
    return None

# 원본 보드와 갱신된 보드를 가져와 새로운 piece가 어디에 두어졌는지 계산 
def detect_location(origin_board, new_board):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if origin_board[row][col] != new_board[row][col]:
                return (row, col)
    return None

# place가능한 location반환
def detect_able_locs(board):
    return [(row, col) for row, col in product(range(4), range(4)) if board[row][col]==0]    

# 승리조건
def check_line(line):
    if 0 in line:
        return False  # Incomplete line
    characteristics = np.array([pieces[piece_idx - 1] for piece_idx in line])
    for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
        if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
            return True
    return False

# 승리조건
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

# 승리조건
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

# 선택한 piece의 정보가 들어있음.
class select_Node():
    def __init__(self, parent, piece, able_pieces, minmax, depth = 0):
        self.parent = parent
        self.board = parent.board
        self.data = piece
        self.able_pieces = able_pieces
        self.childNodes = []
        self.minmax = minmax
        self.depth = depth
        self.x = 0.0
        self.n = 0
        self.clear = 0

    #childNodes확장 (이미 있는것 재외)
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
                        newNode = place_Node(self, newBoard, (row,col), newAblePiece, not self.minmax, self.depth+1)
                        self.childNodes.append(newNode)

    # 뭘 선택했는지 출력 (디버깅용)
    def print_it(self):
        print(self.minmax, ' selected ', toString_mbti(self.data))
                
# piece를 둔 location정보가 들어있음        
class place_Node():
    def __init__(self, parent, board, location, able_pieces, minmax, depth = 0):
        self.parent = parent
        self.board = board
        self.data = location
        self.able_pieces = able_pieces        
        self.childNodes = []
        self.minmax = minmax
        self.depth = depth
        self.n = 0
        self.x = 0.0
        self.clear = 0
    
    #childNodes확장 (이미 있는것 재외)
    def expand(self):
        for piece in self.able_pieces:
            if (not detect_child(self.childNodes, piece)):
                newAblePiece = copy.deepcopy(self.able_pieces)
                newAblePiece.remove(piece)
                newNode = select_Node(self, piece, newAblePiece, self.minmax, self.depth+1)
                self.childNodes.append(newNode)
    
    # 뭘 선택했는지 출력 (디버깅용)
    def print_it(self):
        print(self.minmax, ' placed ', self.data)
        print_board(self.board)

# p2
class P2():
    head = place_Node(None, new_board() , None, new_pieces(), True)
    current_location = None
    current_board = new_board()
    current_piece = None
    def __init__(self, board, available_pieces):  # 생성자
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces

        # 게임이 다시 시작된다면 reset
        if len(available_pieces) == 16:
            P2.reset()

    # 상대가 둘 piece 선정
    def select_piece(self):
        if (P2.head.parent != None):
            head_set([P2.current_location])
        selected_piece = monte_carlo(P2.head)
        print("P2: selected...", selected_piece, toString_mbti(selected_piece))
        P2.current_piece = selected_piece
        P2.current_board = copy.deepcopy(self.board)
        return selected_piece

    # 내가 둘 location 선정
    def place_piece(self, selected_piece):
        head_set([P2.current_piece, detect_location(P2.current_board, self.board), selected_piece])
        position = monte_carlo(P2.head)
        print("P2: placed...", position)
        P2.current_location = position
        return position
    
    def reset():
        P2.head = place_Node(None, new_board() , None, new_pieces(), True)
    
# monte carlo 트리의 head를 갱신해줌. (게임이 진행됨에 따라 필요없는 부모노드를 무시)
def head_set(list):
    for data in list:
        P2.head.expand()
        P2.head = detect_child(P2.head.childNodes, data)
        
        if (not P2.head):
            print(data,"실패")

# monte carlo 진행 (iterate할 노드 선정 -> simulate -> backpropagate)
def monte_carlo(root):
    begin = time.time()

    if (not root.clear):
        while time.time() < begin + CALC_TIME[root.depth]:
            if root.clear:
                print('P2: all possability calculated...!')
                if(root.x == 1):
                    print('P2: will win!')
                else:
                    print('P2: might lose...')
                break

            node = root
            while node.childNodes:
                node = select_to_roll_child(node)

            x = simulate_game(node)
            backpropagate(node, x)

    best_child = select_best_child(root)
    return best_child.data

# iterate할 노드 선정 (ucb1)
def select_to_roll_child(node):
    def ucb1_score(child):
        cleared = 0
        if child.n == 0:
            return float('inf')  # Prioritize unexplored nodes
        
        if child.clear == 1:
            cleared = -999
            
        exploitation = abs(child.x) # 중요! 절댓값!
        exploration = (EXPLORATION_WEIGHT - WEIGHT_CHANGE*child.depth) * np.sqrt(np.log(node.n) / child.n)
        return exploitation + exploration + cleared

    return max(node.childNodes, key=ucb1_score)

# head노드의 자식중 최선의 선택을 고름
def select_best_child(node):
    # print_child_data(node)
    return max(node.childNodes, key=lambda child: child.x + child.clear)

# iteration (simualte)   
# 게임이 끝날때까지 들어간 깊이에 따라 x값이 다름
def simulate_game(node):
    node.expand()
    
    # 이것저것 깊은복사해옴
    temp_board = copy.deepcopy(node.board)
    temp_able_pieces = copy.deepcopy(node.able_pieces)
    temp_able_locs = detect_able_locs(temp_board)
    current_minmax = node.minmax

    #select노드였다면 초반에 추가 진행 + 1
    if (isinstance(node, select_Node)):
        current_minmax = not current_minmax
        row, col = random.choice(temp_able_locs)
        temp_able_locs.remove((row,col))
        temp_board[row][col] = pieces.index(node.data) + 1

    # 끝난 노드인지 확인
    if check_win(temp_board):
        if (current_minmax):
            node.clear = 1
        else:
            node.clear = -1

    if not len(temp_able_pieces):
        node.clear = 0.0001

    # 게임이 끝날때 까지 roll down
    while not check_win(temp_board):
        if (not temp_able_pieces or not temp_able_locs):
            return 0
        
        current_minmax = not current_minmax

        piece = random.choice(temp_able_pieces)
        row, col = random.choice(temp_able_locs)

        temp_able_pieces.remove(piece)
        temp_able_locs.remove((row,col))

        temp_board[row][col] = pieces.index(piece) + 1

    if (current_minmax == node.minmax):
        return 1 # 승리
    else:
        return -1 # 패배

# backpropagate하여 부모노드의 n, x등을 갱신
def backpropagate(node, x):
    temp_x = x

    while node is not None:
        if (not node.minmax):
            temp_x = -x

        adder = 1 if isinstance(node, place_Node) == node.minmax else -1

        if len(node.childNodes):    
            trigger = True   
            for child in node.childNodes:
                if child.clear == adder:
                    node.clear = adder
                    trigger = False
                    break
                elif child.clear != -adder:
                    trigger = False
                    continue

            if trigger:
                node.clear = -adder

        node.x = (node.n * node.x + temp_x) / (node.n + 1)
        node.n += 1
        node = node.parent





