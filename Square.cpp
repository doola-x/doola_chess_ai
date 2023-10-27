#include "Square.hpp"

Square::Square(){
	this->rank = 0;
	this->file = 'z';
}

Square::~Square(){
	delete this;
}

Square::Square(int _rank, char _file){
	this->rank = _rank;
	this->file = _file;
}

std::string Square::getSquare(int coordX, int coordY){
	std::string square;
	float squareDimension = 480.0/8.0;

	// 480 / 8 = 60. 
	//for every 60, rank and file increase 
	int rank[8] = {8, 7, 6, 5, 4, 3, 2, 1};
	char file[8] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};

	float fileIdx = coordX / squareDimension;
	float rankIdx = coordY / squareDimension;

	std::round(fileIdx);
	std::round(rankIdx);

	square += file[(int)fileIdx];
	square += std::to_string(rank[(int)rankIdx]);

	return square;
}