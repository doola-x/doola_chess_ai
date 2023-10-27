#include "app.hpp"
#include "Piece.hpp"

using namespace std;

Piece::Piece(){
	this.alligience = 0;
	this.type = "undefined";
}

Piece::~Piece(){
	delete this;
}

Piece::Piece(int _allegience, string _type){
	this.alligience = _allegience;
	this.type = _type;
}