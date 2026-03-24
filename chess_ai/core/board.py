"""
Board representation utilities.
"""
from __future__ import annotations

import numpy as np
import chess

# ─── Piece encoding ───────────────────────────────────────────────────────────

PIECE_TO_IDX: dict[str, int] = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,   # White
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11, # Black
}
IDX_TO_PIECE: dict[int, str] = {v: k for k, v in PIECE_TO_IDX.items()}

# Board tensor shape: (rows, cols, channels)
# Channels 0-11: one-hot piece occupancy; channel 12: active color (1=white, 0=black)
BOARD_SHAPE = (8, 8, 13)


# ─── Conversion functions ─────────────────────────────────────────────────────

def fen_to_tensor(fen: str) -> np.ndarray:
    """
    Convert a FEN string to an (8, 8, 13) float32 numpy array.

    The tensor is indexed as tensor[rank, file, channel] where rank 0 = rank 1
    (bottom of the board) and file 0 = the a-file.
    """
    board = chess.Board(fen)
    tensor = np.zeros(BOARD_SHAPE, dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = chess.square_rank(square)   # 0–7
            file = chess.square_file(square)   # 0–7
            idx = PIECE_TO_IDX[piece.symbol()]
            tensor[rank, file, idx] = 1.0

    tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    return tensor


def tensor_to_fen(tensor: np.ndarray) -> str:
    """
    Reconstruct an approximate FEN from an (8, 8, 13) tensor.

    Note: castling rights and en passant are not encoded in the tensor,
    so the returned FEN omits them (uses '-').
    """
    board = chess.Board(fen=None)
    board.clear()

    for rank in range(8):
        for file in range(8):
            for idx in range(12):
                if tensor[rank, file, idx] > 0.5:
                    piece = chess.Piece.from_symbol(IDX_TO_PIECE[idx])
                    square = chess.square(file, rank)
                    board.set_piece_at(square, piece)

    board.turn = chess.WHITE if tensor[0, 0, 12] > 0.5 else chess.BLACK
    return board.fen()


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert a live chess.Board object (skips FEN serialisation)."""
    tensor = np.zeros(BOARD_SHAPE, dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[rank, file, PIECE_TO_IDX[piece.symbol()]] = 1.0
    tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    return tensor
