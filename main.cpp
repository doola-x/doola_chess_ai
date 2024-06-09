#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>
#include "Game.hpp"
#include "Square.hpp"

int main() {
    Game game;
	sf::Clock debounceClock;

    while (game.IsRunning())
    {
    	game.Update(debounceClock);
		game.Render();
    }

    return EXIT_SUCCESS;

}
