#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

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
	sf::Texture texture;
	sf::Sprite board_sprite;
	sf::Sprite piece_sprite;
private:
	void InitWindow();
	void HandleEvents(sf::Clock dBounce);
};