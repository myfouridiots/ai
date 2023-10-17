def is_safe(board, row, col, N):
    # Check if there is a queen in the same column
    for i in range(row):
        if board[i][col] == 'Q':
            return False
    # Check if there is a queen in the upper left diagonal
    i, j = row, col
    while i >= 0 and j >= 0:
        if board[i][j] == 'Q':
            return False
        i -= 1
        j -= 1
    # Check if there is a queen in the upper right diagonal
    i, j = row, col
    while i >= 0 and j < N:
        if board[i][j] == 'Q':
            return False
        i -= 1
        j += 1
    return True

def solve_n_queens(board, row, N, solutions):
    if row == N:
        solutions.append(["".join(row) for row in board])
        return
    for col in range(N):
        if is_safe(board, row, col, N):
            board[row][col] = 'Q'
            solve_n_queens(board, row + 1, N, solutions)
            board[row][col] = '.'
            
def n_queens(N):
    board = [['.' for _ in range(N)] for _ in range(N)]
    solutions = []
    solve_n_queens(board, 0, N, solutions)
    return solutions
# Get the number of queens from the user
N = int(input("Enter the number of queens: "))
solutions = n_queens(N)
print(f"Number of solutions for {N}-queens problem: {len(solutions)}")
for solution in solutions:
    for row in solution:
        print(" ".join(row))
    print()