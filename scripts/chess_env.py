import chess
import chess.engine
import subprocess
import random

''' 
====================================================================
chess env planning
====================================================================

need:
	- functions to:
		- begin play (+)
		- step (select move)
			- random move
			- best move (according to policy network) (+)
			- explorative move (not so random?)
		- calculate reward
			- material count (+)
			- king safety (+)*
			- pawn structure
			- center control (+)
			- piece mobility
		- return state
			- current position
			- draw, win, loss
'''

class ChessEnvironment:
	def __init__(self, engine_path='../stockfish'):
		self.board = chess.Board()
		self.engine_path = engine_path
		self.engine = None

	def start_engine(self):
		self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

	def stop_engine(self):
		if self.engine:
			self.engine.quit()

	def reset_board(self):
		# Reset the chess board to the initial state
		self.board.reset()
		color_det = random.random()
		self.inf_is_white = True if color_det > 0.4999 else False

	def make_move(self, player):
		if player == "inference":
			result = subprocess.run(f'/opt/homebrew/bin/python3.11 inference.py "{board.fen()}"', capture_output=True, shell=True)
			print(f"result: {result}")
			if result.stdout:
				move = result.stdout.decode().strip()
			self.board.push_san(move)
		else:
			result = self.engine.play(self.board, chess.engine.Limit(time=0.5))
			self.board.push(result.move)

	def calc_material_count(self):
		white_pawns = len(self.board.pieces(chess.PAWN, chess.WHITE))
		white_rooks = len(self.board.pieces(chess.ROOK, chess.WHITE)) * 5
		white_knights = len(self.board.pieces(chess.KNIGHT, chess.WHITE)) * 3
		white_bishops = len(self.board.pieces(chess.BISHOP, chess.WHITE)) * 3
		white_queen = len(self.board.pieces(chess.QUEEN, chess.WHITE)) * 9
		white_king = 1
		black_pawns = len(self.board.pieces(chess.PAWN, chess.BLACK))
		black_rooks = len(self.board.pieces(chess.ROOK, chess.BLACK)) * 5
		black_knights = len(self.board.pieces(chess.KNIGHT, chess.BLACK)) * 3
		black_bishops = len(self.board.pieces(chess.BISHOP, chess.BLACK)) * 3
		black_queen = len(self.board.pieces(chess.QUEEN, chess.BLACK)) * 9
		black_king = 1
		white_material = white_pawns + white_rooks + white_knights + white_bishops + white_queen + white_king
		black_material = black_pawns + black_rooks + black_knights + black_bishops + black_queen + black_king
		return white_material - black_material

	def king_attackers(self):
		white_king = self.board.king(chess.WHITE)
		black_king = self.board.king(chess.BLACK)
		white_attackers = len(self.board.attackers(chess.WHITE, chess.SQUARES[black_king]))
		black_attackers = len(self.board.attackers(chess.BLACK, chess.SQUARES[white_king]))
		return white_attackers - black_attackers

	def center_attackers(self):
		white_d4 = len(self.board.attackers(chess.WHITE, chess.D4))
		black_d4 = len(self.board.attackers(chess.BLACK, chess.D4))
		white_e4 = len(self.board.attackers(chess.WHITE, chess.E4))
		black_e4 = len(self.board.attackers(chess.BLACK, chess.E4))
		white_d5 = len(self.board.attackers(chess.WHITE, chess.D5))
		black_d5 = len(self.board.attackers(chess.BLACK, chess.D5))
		white_e5 = len(self.board.attackers(chess.WHITE, chess.E5))
		black_e5 = len(self.board.attackers(chess.BLACK, chess.E5))
		white_center = white_d4 + white_d5 + white_e4 + white_e5
		white_center = white_center / 4
		black_center = black_d4 + black_d5 + black_e4 + black_e5
		black_center = black_center / 4
		return white_center - black_center

	def is_legal_move(self, move):
		try:
			self.board.push_san(move)
			self.board.pop()
			return True
		except:
			#self.push_legal_move()
			return False

	def push_legal_move(self):
			moves = list(self.board.legal_moves)
			if not moves:
				return "done"
			random_move = random.choice(moves)
			self.board.push(random_move)

	def step(self, move):
		keys=['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8',
				'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
				'bxa8', 'axb8', 'bxc8', 'cxb8', 'cxd8', 'exd8', 'exf8','fxe8', 'fxg8', 'gxf8', 'gxh8', 'hxg8',
				'bxa1', 'axb1', 'bxc1', 'cxb1', 'cxd1', 'exd1', 'exf1','fxe1', 'fxg1', 'gxf1', 'gxh1', 'hxg1']
		if (move in keys): move = move + '=Q'
		self.board.push_san(move)
		done = self.board.is_game_over()
		if (not done):
			self.make_move('stockfish')
		# calculate reward fns
		material_count = self.calc_material_count() # returns material balance, + for white adv - for black
		king_safety = self.king_attackers() # counts who has more attackers on the other king, + -
		center_control = self.center_attackers() # counts who has more attackers on the center, + -
		weights = {'material': 5.0, 'center': 0.2, 'king_safety': 0.8}
		reward = (weights['material'] * material_count +
	          weights['center'] * center_control +
	          weights['king_safety'] * king_safety)
		#if (legal == False): reward = reward - (reward * .1)
		done = self.board.is_game_over()
		return reward, done

	def __enter__(self):
		self.start_engine()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.stop_engine()

'''
this function should be in the training script actually
def reset():
	board = chess.Board()
	color_det = random.random()
	if (color_det > .4999): # randomly determine if inference bot plays as white or black
		inf_is_white = True
	else:
		inf_is_white = False'''