#include "Square.hpp"
#include <cmath>

Square::Square()
{
	this->rank = '0';
	this->file = 'z';
	this->piece = new Piece();
}

Square::Square(char _file, char _rank){
	this->file = _file;
	this->rank = _rank;
}

Square::~Square(){
	delete this;
}

std::string Square::getSquareFromClick(int coordX, int coordY){
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

std::array<float, 2> Square::getCoordsFromSquare(){
	//here we need to do the reverse of the function above
	// 478 total units in each direction -- starting point at 10,10 for a8
	// so if b8, then we add 478/8 in the x direction
	// if a7, add in the y direction

	std::array<float, 2> result = {10.f, 10.f};

	int rank[8] = {'8', '7', '6', '5', '4', '3', '2', '1'};
	char file[8] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};

	float x = 0.f;
	for (int i = 0; i < 8; i++){
		if (this->rank != rank[i]){
			x += 60.8f;
		}
		else {
			result[1] = x;
			break;
		}
	}
	x = 0.f;
	for (int i = 0; i < 8; i++){
		if (this->file != file[i]){
			x += 60.8f;
		}
		else {
			result[0] = x;
			break;
		}
	}

	return result;
}
