#pragma once

#include <iostream>
#include "Piece.hpp"
#include <array>

class Square {
public:
	Piece* piece;
	char rank;
	char file;
	bool white_vision;
	bool black_vision;

	Square();
	Square(char _file, char _rank);
	~Square();
	
	std::string static getSquareFromClick(int coordX, int coordY);
	std::array<float, 2> getCoordsFromSquare();
};
