#include "Game.cpp"
#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

int main() {

    //create a game
    Game game;
    //create a debounce clock
	sf::Clock debounceClock;  

    while (game.IsRunning())
    {
    	//handles events
    	game.Update(debounceClock);

		//renders screen (eventually this should handle pieces placement)
		game.Render();

    }

    return EXIT_SUCCESS;

}
