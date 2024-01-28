#pragma once
#include <iostream>
class Piece {
public:
	int allegience;
	char type;
	bool selected;

	Piece();
	~Piece();

	Piece(int _allegience, char _type, bool selected);

	char getType();
};