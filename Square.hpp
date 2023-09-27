#include <iostream>
#include "Piece.hpp"

class Square {
public:
	int rank;
	char file;
	Piece piece;

	Square(int _rank, char _file, Piece _piece);
	~Square();

	std::string static getSquare(int coordX, int coordY);
};