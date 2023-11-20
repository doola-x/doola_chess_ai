# planning the game class
	- square array, 64 members
	- game represented in string format -- rnbqkbnrpppppppp00000000000000000000000000000000pppppppprnbqkbnr
	- in our case, we can use a length 63 object array. accessing members of this class can be computed with getSquareFromClick(), although there is probably a better way to do thos
	- the game should be initilized by giving the 1st/2nd and 7th/8th ranks the correct piece types for each square -- 