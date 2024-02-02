#pragma once
#include <iostream>
class Piece {
public:
	int allegience;
	char type;

	Piece();
	~Piece();
	Piece(int _allegience, char _type);

	char getType();
	std::vector<int> getKingMoves(int i);
};