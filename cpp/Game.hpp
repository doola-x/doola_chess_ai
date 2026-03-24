#pragma once
#include "Square.hpp"
#include "Rules.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

class Game {
private:
    std::thread commThread;
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<std::string> moveQueue;
    std::string pythonResponse;
    bool isRunning;
public: 
	Game();
	~Game();

	void Update(sf::Clock dBounce);
	void Render();
	void RenderBoard();
	void RenderPieces();
	void DoMove(int dropSquare, bool send);
	void CommWithPython();
	int SendMoveToPython(const std::string& move);
	void ProcessPythonResponse(const std::string& response);
	bool IsRunning();

	sf::RenderWindow game_window;
	sf::Event game_event;
	sf::Texture board_texture;
	sf::Texture pawn_texture;
	sf::Texture b_pawn_texture;
	sf::Texture rook_texture;
	sf::Texture b_rook_texture;
	sf::Texture knight_texture;
	sf::Texture b_knight_texture;
	sf::Texture bishop_texture;
	sf::Texture b_bishop_texture;
	sf::Texture king_texture;
	sf::Texture b_king_texture;
	sf::Texture queen_texture;
	sf::Texture b_queen_texture;
	sf::Sprite board_sprite;
	sf::Sprite pawn_sprite;
	sf::Sprite b_pawn_sprite;
	sf::Sprite rook_sprite;
	sf::Sprite b_rook_sprite;
	sf::Sprite knight_sprite;
	sf::Sprite b_knight_sprite;
	sf::Sprite bishop_sprite;
	sf::Sprite b_bishop_sprite;
	sf::Sprite king_sprite;
	sf::Sprite b_king_sprite;
	sf::Sprite queen_sprite;
	sf::Sprite b_queen_sprite;	
	Square* allSquares[64];
	Rules* rules;
	int selectedSquare;
	int sock;
	std::string sqr_c;
	std::string sqr_r;
	std::string move_str;
	bool isUserTurn;

private:
	void InitWindow();
	void InitGame();
	void InitSocket();
	void HandleEvents(sf::Clock dBounce);
};