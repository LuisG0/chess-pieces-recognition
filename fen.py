import numpy as np

PIECE_TYPES = ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K','P','_']


def fen2board(fen):
    rows = fen.split(sep='/')

    if len(rows) != 8:
        raise ValueError(f"fen must have 8 rows: {fen}")

    board = []
    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend(['_'] * int(char))
            else:
                board_row.append(char)
        if len(board_row) != 8:
            raise ValueError(f"Each fen row must have 8 positions: {fen}")
        board.append(board_row)

    return board


def compareFen(fen1, fen2):
    confussionMatrix = np.zeros((len(PIECE_TYPES),len(PIECE_TYPES)),np.uint8)

    predictedBoard = fen2board(fen1)
    board = fen2board(fen2)
    count = 0
    for i in range(8):
        for j in range(8):
            confussionMatrix[PIECE_TYPES.index(board[i][j]),PIECE_TYPES.index(predictedBoard[i][j])] +=1
            if predictedBoard[i][j] != board[i][j]:
                count += 1
    accu = (64-count)/64
    return accu,confussionMatrix

def board2fen(board):
    fen = []
    for row in board:
        prev_empty = False
        empty = 0
        for square in row:
            if square == '_':
                empty += 1
                prev_empty = True
            else:
                if prev_empty:
                    prev_empty = False
                    fen.append(str(empty))
                    empty = 0
                fen.append(square)

        if prev_empty:
            fen.append(str(empty))

        fen.append('/')

    return ''.join(fen[:-1])  # Remove final /