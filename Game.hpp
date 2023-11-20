#pragma once
#include "Square.hpp"

class Game {
public: 
	Game();
	~Game();

	void Update(sf::Clock dBounce);
	void Render();
	void RenderBoard();
	void RenderPieces();

	bool IsRunning();

	sf::RenderWindow game_window;
	sf::Event game_event;
	sf::Texture board_texture;
	sf::Texture pawn_texture;
	sf::Texture rook_texture;
	sf::Sprite board_sprite;
	sf::Sprite piece_sprite;
	sf::Sprite rook_sprite;
	Square* allSquares[64];

private:
	void InitWindow();
	void InitGame();
	void HandleEvents(sf::Clock dBounce);
};