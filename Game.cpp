#include "Game.hpp"
#include "Square.cpp"

Game::Game() {
	InitWindow();
}

Game::~Game(){
	//destroy something
}

void Game::HandleEvents(sf::Clock dBounce){
	//do some handling
	sf::Time elapsed = dBounce.restart();
	while (game_window.pollEvent(game_event))
	{
	    // check the type of the event...
	    switch (game_event.type)
	    {
	        // window closed
	        case sf::Event::Closed:
	            game_window.close();
	            break;

	        // mouse click
	        case sf::Event::MouseButtonPressed:
	            if (game_event.mouseButton.button == sf::Mouse::Left) {
	                sf::Vector2i mousePos = sf::Mouse::getPosition(game_window);
	                elapsed = dBounce.restart();

	                std::cout << "Clicked at Sqaure: (" << Square::getSquare(mousePos.x, mousePos.y)<< ")" << std::endl;
	                std::cout << "Mouse at: (" << mousePos.x << "," << mousePos.y << ")"<< std::endl;
	            }
	        // mouse released
            case sf::Event::MouseButtonReleased:
                if (game_event.mouseButton.button == sf::Mouse::Left && elapsed.asSeconds() > 0.001) {
                    sf::Vector2i releasePos = sf::Mouse::getPosition(game_window);

                    std::cout << "Mouse Released at Sqaure: (" << Square::getSquare(releasePos.x, releasePos.y)<< ")" << std::endl;
                    std::cout << "Debounce clock time at: " << elapsed.asSeconds() << std::endl;
                }

	        // we don't process other types of events
	        default:
	            break;
	    }
	}
}

void Game::Update(sf::Clock dBounce) {
	//do some updating
	HandleEvents(dBounce);
}

void Game::Render() {
	       // clear the window with black color
        game_window.clear(sf::Color::Black);

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

        game_window.draw(text);

        sf::Texture texture;
		if (!texture.loadFromFile("images/board2.png"))
		{
		    // error...
		}

		sf::Sprite sprite;
		sprite.setTexture(texture);
		sprite.setPosition(sf::Vector2f(10.f, 10.f)); // absolute position
		game_window.draw(sprite);
        // end the current frame

        game_window.display();
}

void Game::InitWindow() {
	game_window.create(sf::VideoMode(800, 500), "doola chess");
}

bool Game::IsRunning() {
	return game_window.isOpen();
}


