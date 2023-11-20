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