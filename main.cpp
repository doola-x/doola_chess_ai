#include "Square.cpp"
#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

int main() {

    // Create a window
    sf::RenderWindow window(sf::VideoMode(800, 500), "doola chess");
	sf::Clock debounceClock;  

    while (window.isOpen())
    {      // timer to measure time intervals
    	// check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
		// while there are pending events...
		while (window.pollEvent(event))
		{
			sf::Time elapsed = debounceClock.restart();
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
		                elapsed = debounceClock.restart();

		                std::cout << "Mouse Clicked at Sqaure: (" << Square::getSquare(mousePos.x, mousePos.y)<< ")" << std::endl;
		                std::cout << "Debounce clock time at: " << elapsed.asSeconds() << std::endl;
		            }
		        // mouse released
	            case sf::Event::MouseButtonReleased:
	                if (event.mouseButton.button == sf::Mouse::Left && elapsed.asSeconds() > 0.01) {
	                    sf::Vector2i releasePos = sf::Mouse::getPosition(window);

	                    std::cout << "Mouse Released at Sqaure: (" << Square::getSquare(releasePos.x, releasePos.y)<< ")" << std::endl;
	                    std::cout << "Debounce clock time at: " << elapsed.asSeconds() << std::endl;
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
