#pragma once
#include "Square.hpp"

class Rules {
public:
	Rules();
	~Rules();

	int isValidMove(char type, int dropSquare, int selectedSquare, char takes, Square* allSquares[64]);
	int isValidPawnMove(int diff, char takes);
	int isValidKnightMove(int diff, Square* allSquares[64]);
	int isValidRookMove(int selectedSquare, int dropSquare, int diff, Square* allSquares[64]);
};