#include <iostream>
#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

std::string getSquare(int coordX, int coordY){
	std::string square;
	int squareDimension = 480/8;

	// 480 / 8 = 60. 
	//for every 60, rank and file increase 
	int rank[8] = {8, 7, 6, 5, 4, 3, 2, 1};
	char file[8] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};

	int fileIdx = coordX / squareDimension;
	char rankIdx = coordY / squareDimension;

	square += file[fileIdx];
	square += std::to_string(rank[rankIdx]);

	return square;
}

int main() {

    // Create a window
    sf::RenderWindow window(sf::VideoMode(800, 500), "doola chess");
        // run the program as long as the window is open

    while (window.isOpen())
    {
    	// check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
		// while there are pending events...
		while (window.pollEvent(event))
		{
		    // check the type of the event...
		    switch (event.type)
		    {
		        // window closed
		        case sf::Event::Closed:
		            window.close();
		            break;

		        // mouse click
		        case sf::Event::MouseButtonPressed:
		            if (event.mouseButton.button == sf::Mouse::Left) {
		                sf::Vector2i mousePos = sf::Mouse::getPosition(window);
		                std::cout << "Mouse Clicked at Position and Square: (" << mousePos.x << ", " << mousePos.y << ")" << std::endl;
		                std::cout << getSquare(mousePos.x, mousePos.y) << std::endl;
		            }
		        // mouse released
	            case sf::Event::MouseButtonReleased:
	                if (event.mouseButton.button == sf::Mouse::Left) {
	                    sf::Vector2i releasePos = sf::Mouse::getPosition(window);
	                    std::cout << "Mouse Released at Position and Sqaure: (" << releasePos.x << ", " << releasePos.y << ")" << std::endl;
	                    std::cout << getSquare(releasePos.x, releasePos.y) << std::endl;
	                }

		        // we don't process other types of events
		        default:
		            break;
		    }
		}

        // clear the window with black color
        window.clear(sf::Color::Black);

        sf::Font font;
        if (!font.loadFromFile("fonts/GamiliaDemoRegular-d9DL7.ttf"))
        {
        	//error...
        }

        sf::Text text;
        text.setFont(font);
        text.setString("doola chess bot");
        text.setCharacterSize(16);
        text.setFillColor(sf::Color::White);
        text.setStyle(sf::Text::Bold | sf::Text::Underlined);
        text.setPosition(sf::Vector2f(500.f, 1.f));

        window.draw(text);

        sf::Texture texture;
		if (!texture.loadFromFile("images/board2.png"))
		{
		    // error...
		}

		sf::Sprite sprite;
		sprite.setTexture(texture);
		sprite.setPosition(sf::Vector2f(10.f, 10.f)); // absolute position
		window.draw(sprite);
        // end the current frame
        window.display();

    }
    return 0;

}
