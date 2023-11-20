#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include "Game.hpp"
#include "Square.hpp"

int main() {

    //create a game
    Game game;
    //create a debounce clock
	sf::Clock debounceClock;

    while (game.IsRunning())
    {
    	game.Update(debounceClock);
		game.Render();
    }

    return EXIT_SUCCESS;

}
