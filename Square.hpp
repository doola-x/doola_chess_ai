#include <iostream>
#include "Piece.hpp"

using namespace std;

class Square {
public:
	int rank;
	char file;

	Square();
	~Square();

	Square(int _rank, char _file);
	
	string static getSquare(int coordX, int coordY);
};