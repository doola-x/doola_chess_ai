# todo

implement user castling in play_inference python script (+)

run tactics training (+)

	- tactics training overcompensates for trying to find tactical moves, wild sacrifices constantly
	- new strategy is just once a month to fetch new games and fine tune on those, maybe a seperate tactics model when we think a tactic is present? it could be determined with the new tensor style described in etc/plans.md / etc/thoughts.md. when these values are present, we value the tactics models input more and run calculations with that in mind

do move castling (?)

ambiguous rook/knight movement output from gui (+)

graceful error handling on illegal move suggestion (+)

rules engine (?)

	- create a new class rules to host functions and calculate rules for all pieces: 
		- pawn (+)
		- rook (+)
		- knight (+)
		- bishop ()
		- queen ()
		- king ()
	- when a user clicks while it is not their turn, no action is taken (+)
	- when a user clicks and releases on the same square, no action is taken (+)

add support to play as black ( )
	
	- user selects black or white before play begins
	- on selection, the piece rendering would also need to be flipped
	- as simple as 