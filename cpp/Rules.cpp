#include "Rules.hpp"
#include "Square.hpp"

Rules::Rules(){
	//construct something?
}

Rules::~Rules(){
	//destroy something?
}

// to do : pin check? or just allow king to be captured?
	// implement a checkmate check
	// if no pin check, king capture logic as well

int Rules::isValidMove(char type, int dropSquare, int selectedSquare, char takes, Square* allSquares[64]) {
	if (allSquares[dropSquare]->piece->allegience == allSquares[selectedSquare]->piece->allegience) {
		return -1;
	}
	int diff = dropSquare - selectedSquare;
	if (type == 'p'){
		int i = isValidPawnMove(diff, takes);
		return i;
	}
	if (type == 'n'){
		int i = isValidKnightMove(diff, allSquares);
		return i;
	}
	if (type == 'r'){
		int i = isValidRookMove(selectedSquare, dropSquare, diff, allSquares);
		return i;
	}
	if (type == 'b'){
		int i = isValidBishopMove(selectedSquare, dropSquare, diff, allSquares);
		return i;
	}
	if (type == 'q'){
		int i = isValidRookMove(selectedSquare, dropSquare, diff, allSquares);
		int j = isValidBishopMove(selectedSquare, dropSquare, diff, allSquares);
		if (i == 1 || j == 1) {
			return 1;
		} else {
			return -1;
		}
	}
	if (type == 'k'){
		//king move check
	}
	return 0;
}

int Rules::isValidRookMove(int selectedSquare, int dropSquare, int diff, Square* allSquares[64]) {
	//go from selected square to drop square in rook movement pattern (which is?) and confirm n
	std::cout << "rook square: " << selectedSquare << ", drop square: " << dropSquare << std::endl;
	if ((diff % 8 == 0) || ((0 < diff) && (diff < 8)) || (diff > -8) && (diff < 0)){
		if (diff % 8 == 0) {
			// selected square is rook being moved, drop square is empty or piece being captured
			int mult = (diff > 0) ? 1 : -1;
			selectedSquare = selectedSquare + (8 * mult);
			while (selectedSquare != dropSquare) {
				if (allSquares[selectedSquare]->piece->type != 'u'){
					return -1;
				}
				selectedSquare = selectedSquare + (8 * mult);
			}
			return 1;
		} else {
			int mult = (diff > 0) ? 1 : -1;
			selectedSquare = selectedSquare + (1 * mult);
			while (selectedSquare != dropSquare){
				if (allSquares[selectedSquare]->piece->type != 'u'){
					return -1;
				}
				selectedSquare = selectedSquare + (1 * mult);
			}
			return 1;
		}
	} else {
		return -1;
	}
}

int Rules::isValidBishopMove(int selectedSquare, int dropSquare, int diff, Square* allSquares[64]) {
	if (diff % 9 != 0 && diff % 7 != 0) { 
		return -1;
	}
	int sqrMover = 0;
	if (diff % 9 == 0) {
		sqrMover = 9;
	}
	if (diff % 7 == 0) {
		sqrMover = 7;
	}
	int mult = diff > 0 ? 1 : -1;
	selectedSquare = selectedSquare + (sqrMover * mult);
	while (selectedSquare != dropSquare) {
		if (allSquares[selectedSquare]->piece->type != 'u'){
			return -1;
		}
		selectedSquare = selectedSquare + (sqrMover * mult);
	}

	return 1;
}

int Rules::isValidKnightMove(int diff, Square* allSquares[64]) {
	if (diff == 6 || diff == 10 || diff == 15 || diff == 17 || diff == -6 || diff == -10 || diff == -15 || diff == -17) {
		return 1;
	} else {
		return -1;
	}
}

int Rules::isValidPawnMove(int diff, char takes) {
	if (diff == 8 || diff == 16 || diff == 7 || diff == 9) {
		if (diff == 7 || diff == 9){
			if (takes == 'u'){
				//to do: en passant check
				return -1;
			} else {
				return 1;
			}
		}
		if (diff == 8) {
			if (takes == 'u') {
				return 1;
			} else {
				return -1;
			}
		}
		return 1;
	} else {
		return -1;
	}
}
