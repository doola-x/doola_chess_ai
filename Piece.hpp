#include <iostream>
using namespace std;

class Piece {
public:
	int allegience;
	string type;

	Piece();
	~Piece();

	Piece(int _allegience, string type);
};