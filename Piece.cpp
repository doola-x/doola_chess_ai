#include "Piece.hpp"

Piece::Piece(){
	this->allegience = 0;
	this->type = 'u';
}

Piece::Piece(int _allegience, char _type){
	this->allegience = _allegience;
	this->type = _type;
}
char Piece::getType() {
	return this->type;
}

Piece::~Piece(){
	delete this;
}

std::vector<int> getKingMoves(int i){
	//passed position of king, return positions in all 8 direction
	//  add a row, subtract one -- if in the board, add it to possible moves, for now, edges of boards will be in play. fix l8r
	std::vector<int> moves;
	int rowUp = i + 8;
	int rowDown = i - 8;

	//up row
	if (rowUp < 64){
		moves.push_back(rowUp);
		//diagonal
		moves.push_back(rowUp - 1);
		moves.push_back(rowUp + 1);
	}
	//down row
	if (rowDown >= 0){
		moves.push_back(rowDown);
		//diagonal
		moves.push_back(rowDown - 1);
		moves.push_back(rowDown + 1);
	}

	return moves;
}