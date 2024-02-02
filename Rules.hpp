#pragma once
#import "Square.hpp"

class Rules{
public:
	int wKingPos;
	int bKingPos;
	int wKingCheck;
	int bKingCheck;

	Rules();
	~Rules();

	int isCheckmate(Square** allSquares, int size);

};