#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

int main() {

    // Create a window
    sf::RenderWindow window(sf::VideoMode(800, 500), "doola chess");
        // run the program as long as the window is open
    while (window.isOpen())
    {
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // clear the window with black color
        window.clear(sf::Color::Black);

        sf::Font font;
        if (!font.loadFromFile("GamiliaDemoRegular-d9DL7.ttf"))
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
		if (!texture.loadFromFile("board2.png"))
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