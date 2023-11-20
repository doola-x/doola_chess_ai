#pragma once
#include <iostream>
class Piece {
public:
	int allegience;
	char type;

	Piece();
	~Piece();

	Piece(int _allegience, char type);

	char getType();
};