#pragma once

#include <iostream>
#include "Piece.hpp"

class Square {
public:
	char rank;
	char file;
	Piece* piece;

	Square();
	Square(char _file, char _rank);
	~Square();
	
	std::string static getSquareFromClick(int coordX, int coordY);

	std::array<float, 2> getCoordsFromSquare();

};