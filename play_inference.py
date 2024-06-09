import socket
import subprocess
import chess

def process_move(move_str, board):
    # Here you should integrate real move validation logic
    print('in process move')
    try:
        result = board.push_san(move_str)
        print('move pushed')
        return result
    except ValueError:
        return "invalid"

def play_game(process):
    board = chess.Board()
    isUserTurn = True

    host = '127.0.0.1'
    port = 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print("Game server started. Waiting for connection...")
        conn, addr = s.accept()
        print('Connected by', addr)
        with conn:
            while not board.is_checkmate():
                if isUserTurn:
                    data = conn.recv(1024)
                    if not data:
                        break
                    print(f'data recieved: {data}')
                    move_str = data.decode()
                    response = process_move(move_str, board)
                    print(f'response processed: {response}')
                    if (response != "invalid"):
                        isUserTurn = False
                else:
                    fen = board.fen()
                    result = subprocess.run(f'/opt/homebrew/bin/python3.11 inference.py "{fen}"', capture_output=True, shell=True, cwd='scripts')
                    if result.stdout:
                        m = result.stdout.decode().strip()
                        move_valid = process_move(m, board)
                        if (move_valid != "invalid"):
                            m = move_valid.uci()
                            conn.sendall(m.encode())
                            isUserTurn = True

if __name__ == '__main__':
    build_path = './build'
    process = subprocess.Popen(build_path, stdin=subprocess.PIPE, text=True)
    play_game(process)
