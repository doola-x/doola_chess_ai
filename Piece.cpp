#include "Piece.hpp"

Piece::Piece(){
	this->allegience = 0;
	this->type = 'u';
	this->selected = false;
}

Piece::Piece(int _allegience, char _type, bool _selected){
	this->allegience = _allegience;
	this->type = _type;
	this->selected = _selected;
}
char Piece::getType() {
	return this->type;
}

Piece::~Piece(){
	delete this;
}