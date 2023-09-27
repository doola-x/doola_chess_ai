#include "Square.hpp"

std::string Square::getSquare(int coordX, int coordY){
	std::string square;
	int squareDimension = 480/8;

	// 480 / 8 = 60. 
	//for every 60, rank and file increase 
	int rank[8] = {8, 7, 6, 5, 4, 3, 2, 1};
	char file[8] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};

	int fileIdx = coordX / squareDimension;
	char rankIdx = coordY / squareDimension;

	square += file[fileIdx];
	square += std::to_string(rank[rankIdx]);

	return square;
}