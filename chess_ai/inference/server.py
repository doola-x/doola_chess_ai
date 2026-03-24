"""
Socket server — receives moves from the C++ GUI, responds with AI moves.

Protocol:
  - Client sends: FEN string terminated by newline
  - Server responds: UCI move string terminated by newline
  - Client sends "quit" to disconnect

Run:
    python run_server.py --config configs/default.yaml
"""
from __future__ import annotations

import socket
import threading
from typing import Optional

import chess

from chess_ai.config import Config
from chess_ai.inference.engine import InferenceEngine


class ChessServer:
    """
    TCP server wrapping InferenceEngine.

    Parameters
    ----------
    cfg         : Config
    policy_ckpt : Path to policy checkpoint
    value_ckpt  : Optional value checkpoint
    host        : Bind address
    port        : TCP port
    """

    def __init__(
        self,
        cfg: Config,
        policy_ckpt: str,
        value_ckpt: Optional[str] = None,
        host: str = "localhost",
        port: int = 65432,
        temperature: float = 1.0,
        top_k: int = 5,
        value_weight: float = 0.0,
    ):
        self.host = host
        self.port = port
        self.engine = InferenceEngine(
            cfg, policy_ckpt, value_ckpt,
            temperature=temperature,
            top_k=top_k,
            value_weight=value_weight,
        )
        print(f"Chess server ready on {host}:{port}")

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen()
            print(f"Listening on {self.host}:{self.port} …")
            while True:
                conn, addr = srv.accept()
                t = threading.Thread(target=self._handle, args=(conn, addr), daemon=True)
                t.start()

    def _handle(self, conn: socket.socket, addr) -> None:
        print(f"Connection from {addr}")
        with conn:
            buf = ""
            while True:
                chunk = conn.recv(1024).decode("utf-8", errors="replace")
                if not chunk:
                    break
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.lower() == "quit":
                        return

                    response = self._respond(line)
                    conn.sendall((response + "\n").encode("utf-8"))

    def _respond(self, fen: str) -> str:
        try:
            board = chess.Board(fen)
        except Exception:
            return "error:invalid_fen"

        move = self.engine.best_move(board)
        if move is None:
            return "error:no_move"
        return move.uci()
