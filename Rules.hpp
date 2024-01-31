#pragma once
#import "Square.hpp"

class Rules{
public:
	int members;

	Rules();
	~Rules();

	int isCheckmate(Square* allSquares);

};