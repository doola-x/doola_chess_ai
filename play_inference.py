import json
import socket
import subprocess

import chess
import numpy as np
import torch

from chess_ai.config import ModelConfig
from chess_ai.models.nets import load_policy

MODEL_PATH = "models/policy/policy_best.pth"
MOVES_FILE = "data/moves0.json"


def build_idx_to_uci(moves_file: str) -> dict[int, str]:
    with open(moves_file) as f:
        uci_to_idx: dict[str, int] = json.load(f)
    return {v: k for k, v in uci_to_idx.items()}


def fen_to_tensor(fen: str) -> torch.Tensor:
    piece_to_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }
    board_tensor = np.zeros((8, 8, 13), dtype=np.float32)
    pieces, active_color, *_ = fen.split(' ')
    for i, row in enumerate(pieces.split('/')):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                board_tensor[i, col, piece_to_idx[char]] = 1
                col += 1
    board_tensor[:, :, 12] = 1.0 if active_color == 'w' else 0.0
    return torch.from_numpy(board_tensor)


def suggest_move(fen: str, model: torch.nn.Module, idx_to_uci: dict[int, str]) -> str:
    board = chess.Board(fen)
    tensor = fen_to_tensor(fen).unsqueeze(0)  # (1, 8, 8, 13)

    with torch.no_grad():
        logits = model(tensor)[0]  # (num_moves,)

    # Pick the highest-scoring legal move
    order = logits.argsort(descending=True).tolist()
    for idx in order:
        uci = idx_to_uci.get(idx)
        if uci is None:
            continue
        try:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                return uci
        except ValueError:
            continue
    return ""


def process_user_move(move_str: str, board: chess.Board):
    try:
        board.push_san(move_str)
        return True
    except ValueError:
        return False


def play_game(model: torch.nn.Module, idx_to_uci: dict[int, str]):
    board = chess.Board()
    is_user_turn = True

    host = '127.0.0.1'
    port = 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print("Game server started. Waiting for connection...")
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        with conn:
            while not board.is_game_over():
                if is_user_turn:
                    data = conn.recv(1024)
                    if not data:
                        break
                    move_str = data.decode().strip()
                    print(f"User move: {move_str}")
                    if process_user_move(move_str, board):
                        is_user_turn = False
                    else:
                        print("Invalid move, waiting for retry")
                else:
                    uci = suggest_move(board.fen(), model, idx_to_uci)
                    if not uci:
                        print("Model could not find a legal move")
                        break
                    board.push_uci(uci)
                    print(f"AI move: {uci}")
                    conn.sendall(uci.encode())
                    is_user_turn = True


if __name__ == '__main__':
    print(f"Loading model from {MODEL_PATH} ...")
    cfg = ModelConfig()
    model = load_policy(MODEL_PATH, cfg)
    idx_to_uci = build_idx_to_uci(MOVES_FILE)
    print("Model ready.")
    play_game(model, idx_to_uci)
