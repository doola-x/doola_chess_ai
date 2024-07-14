class Node:
	__slots__ = ['fen', 'to_move', 'parent_visits', 'node_visits', 'win_score', 'prev_move', 'next_moves']

	def __init__(self, fen, to_move, parent_visits, win_score=None, prev_move=None, next_moves=None, state_eval=None):
		self.fen = fen
		self.to_move = to_move
		self.parent_visits = parent_visits
		self.node_visits = node_visits
		self.state_eval = state_eval
		self.prev_move = prev_move
		self.next_moves = next_moves if next_move is not None else []

	def uct(total_visits, win_score, node_visits):
		if node_visit_count == 0:
	        return float('inf')
	    return (node_win_score / node_visit_count) + 1.41 * math.sqrt(math.log(total_visits) / node_visit_count)