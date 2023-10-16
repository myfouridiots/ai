
# 1) A. DFS
'''
graph1 = {
'A': set(['B', 'C']),
'B': set(['A', 'D', 'E']),
'C': set(['A', 'F']),
'D': set(['B']),
'E': set(['B', 'F']),
'F': set(['C', 'E'])
}
def dfs(graph, node, visited):
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            dfs(graph,n,visited)
    return visited 
visited = dfs(graph1,'A', [])
print(visited)
'''







# 1) B. BFS
''' 
def bfs(start): 
    queue = [start]
    levels = {start: 0}  # This Dict Keeps track of levels
    visited = set([start])
    while queue:
        node = queue.pop(0)
        neighbours = graph[node] 
        for neighbor in neighbours: 
            if neighbor not in visited: 
                queue.append(neighbor)
                visited.add(neighbor)
                levels[neighbor] = levels[node] + 1
    print(levels)  # print graph level
    return visited 
def bfs_paths(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        vertex, path = queue.pop(0)
        for next_node in graph[vertex] - set(path):
            if next_node == goal:
                yield path + [next_node]
            else:
                queue.append((next_node, path + [next_node]))
    result = list(bfs_paths(graph, 'A', 'F'))
    print(result)  # [['A', 'C', 'F'], ['A', 'B', 'E', 'F']]
def shortest_path(graph, start, goal):
    try:
        return next(bfs_paths(graph, start, goal))
    except StopIteration:
        return None
graph = {
    'A': set(['B', 'C']),
    'B': set(['A', 'D', 'E']),
    'C': set(['A', 'F']),
    'D': set(['B']),
    'E': set(['B', 'F']),
    'F': set(['C', 'E'])
}
result1 = shortest_path(graph, 'A', 'F')
print(result1)  # ['A', 'C', 'F']
print(bfs('A'))  # print graph node
'''







#2) A. Tower of hanoi
'''
def tower_of_hanoi(n, source, auxiliary, target):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    tower_of_hanoi(n-1, source, target, auxiliary)
    print(f"Move disk {n} from {source} to {target}")
    tower_of_hanoi(n-1, auxiliary, source, target)

tower_of_hanoi(3, 'A', 'B', 'C')
'''










#2) B. 8 queen Problem
'''
#bard  only for 4
def is_safe(board, row, col):
    for i in range(4):
        if board[row][i] == 1 or board[i][col] == 1:
            return False

    for i in range(1, 4):
        if row + i < 4 and col + i < 4 and board[row + i][col + i] == 1:
            return False
        if row - i >= 0 and col - i >= 0 and board[row - i][col - i] == 1:
            return False

    return True


def solve_4_queens(board, row):
    if row == 4:
        return True

    for col in range(4):
        if is_safe(board, row, col):
            board[row][col] = 1
            if solve_4_queens(board, row + 1):
                return True
            board[row][col] = 0
    return False


def main():
    board = [[0 for _ in range(4)] for _ in range(4)]

    if solve_4_queens(board, 0):
        print("Solution:")
        for row in board:
            print(row)
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()
















#chatgpt universal
def is_safe(board, row, col):
    # Check if there is a queen in the same column
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper diagonal on left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check upper diagonal on right side
    for i, j in zip(range(row, -1, -1), range(col, len(board))):
        if board[i][j] == 1:
            return False

    return True

def solve_queens(board, row):
    n = len(board)

    if row == n:
        return True

    for col in range(n):
        if is_safe(board, row, col):
            board[row][col] = 1

            if solve_queens(board, row + 1):
                return True

            board[row][col] = 0

    return False

def print_board(board):
    n = len(board)
    for i in range(n):
        for j in range(n):
            print(board[i][j], end=" ")
        print()

# Initialize a 4x4 chessboard
n = 4
chessboard = [[0 for _ in range(n)] for _ in range(n)]

if solve_queens(chessboard, 0):
    print("Solution found:")
    print_board(chessboard)
else:
    print("No solution exists.")


'''










#3) A. to implement alpha beta search.
'''
tree = [[[5, 1, 2], [8, -8, -9]], [[9, 4, 5], [-3, 4, 3]]]
root = 0
pruned = 0

def children(branch, depth, alpha, beta):
    global tree
    global root
    global pruned
    i = 0
    for child in branch:
        if type(child) is list:
            (nalpha, nbeta) = children(child, depth + 1, alpha, beta)
            if depth % 2 == 1:
                beta = nalpha if nalpha < beta else beta
            else:
                alpha = nbeta if nbeta > alpha else alpha
            branch[i] = alpha if depth % 2 == 0 else beta
            i += 1
        else:
            if depth % 2 == 0 and alpha < child:
                alpha = child
            if depth % 2 == 1 and beta > child:
                beta = child
            if alpha >= beta:
                pruned += 1
                break
    if depth == root:
        tree = alpha if root == 0 else beta
    return (alpha, beta)

def alphabeta(in_tree=tree, start=root, upper=-15, lower=15):
    global tree
    global pruned
    global root
    (alpha, beta) = children(tree, start, upper, lower)
    if __name__ == "__main__":
        print("(alpha, beta): ", alpha, beta)
        print("Result: ", tree)
        print("Times pruned: ", pruned)
    return (alpha, beta, tree, pruned)

if __name__ == "__main__":
    alphabeta(None)
'''






# 3) b. Hill CLimb problem
'''
import math

increment = 0.1
startingPoint = [1, 1]
point1 = [1, 5]
point2 = [6, 4]
point3 = [5, 2]
point4 = [2, 1]


def distance(x1, y1, x2, y2):
    dist = math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2)
    return dist


def sumOfDistances(x1, y1, px1, py1, px2, py2, px3, py3, px4, py4):
    d1 = distance(x1, y1, px1, py1)
    d2 = distance(x1, y1, px2, py2)
    d3 = distance(x1, y1, px3, py3)
    d4 = distance(x1, y1, px4, py4)
    return d1 + d2 + d3 + d4


def newDistance(x1, y1, point1, point2, point3, point4):
    d1 = [x1, y1]
    d1temp = sumOfDistances(x1, y1, point1[0], point1[1], point2[0], point2[1],
                             point3[0], point3[1], point4[0], point4[1])
    d1.append(d1temp)
    return d1


def newPoints(minimum, d1, d2, d3, d4):
    if d1[2] == minimum:
        return [d1[0], d1[1]]
    elif d2[2] == minimum:
        return [d2[0], d2[1]]
    elif d3[2] == minimum:
        return [d3[0], d3[1]]
    elif d4[2] == minimum:
        return [d4[0], d4[1]]


minDistance = sumOfDistances(startingPoint[0], startingPoint[1],
                             point1[0], point1[1], point2[0], point2[1],
                             point3[0], point3[1], point4[0], point4[1])

flag = True

i = 1
while flag:
    d1 = newDistance(startingPoint[0] + increment, startingPoint[1], point1, point2,
                     point3, point4)
    d2 = newDistance(startingPoint[0] - increment, startingPoint[1], point1, point2,
                     point3, point4)
    d3 = newDistance(startingPoint[0], startingPoint[1] + increment, point1, point2,
                     point3, point4)
    d4 = newDistance(startingPoint[0], startingPoint[1] - increment, point1, point2,
                     point3, point4)

    print(i, ' ', round(startingPoint[0], 2), round(startingPoint[1], 2))
    minimum = min(d1[2], d2[2], d3[2], d4[2])
    if minimum < minDistance:
        startingPoint = newPoints(minimum, d1, d2, d3, d4)
        minDistance = minimum
        i += 1
    else:
        flag = False
'''













# 4) A. Implmenting the A* algorithm
'''
#steps:- on windows search "%LocalAppData%" file explorer opens, scroll down -> open Programs folder -> python folder ->python310 -> script open cmd there 
# example:- C:\Users\RITIK\AppData\Local\Programs\Python\Python310\Scripts>   (and run the below commands)
# pip install simpleai
# pip install pydot flask

from simpleai.search import SearchProblem, astar
GOAL = 'HELLO WORLD'
class HelloProblem(SearchProblem):
    def actions(self, state):
        if len(state) < len(GOAL):
            return list('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
        else:
            return []

    def result(self, state, action):
        return state + action

    def is_goal(self, state):
        return state == GOAL

    def heuristic(self, state):
        wrong = sum(1 for i in range(len(state)) if state[i] != GOAL[i])
        missing = len(GOAL) - len(state)
        return wrong + missing

problem = HelloProblem(initial_state='')
result = astar(problem)

print("Result state:", result.state)
print("Path to goal:", result.path())'''



