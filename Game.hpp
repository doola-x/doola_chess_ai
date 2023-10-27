#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

class Game {
public: 
	Game();
	~Game();

	void Update(sf::Clock dBounce);
	void Render();

	bool IsRunning();

	sf::RenderWindow game_window;
	sf::Event game_event;
private:
	void InitWindow();
	void HandleEvents(sf::Clock dBounce);
};