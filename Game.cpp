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

void Game::RenderBoard() {
	//do some board rendering
	if (!texture.loadFromFile("images/board2.png"))
	{
	    // error...
	}

	board_sprite.setTexture(texture);
	board_sprite.setPosition(sf::Vector2f(10.f, 10.f)); // absolute position
	game_window.draw(board_sprite);
}


void Game::RenderPieces() {
	//do some piece rendering
	if (!texture.loadFromFile("images/tatiana/pw.png"))
	{
	    // error...
	}
	float startX = 3.5f;

	for (int i = 0; i < 8; i++){
		piece_sprite.setTexture(texture);
		piece_sprite.setScale(.4f, .4f);
		piece_sprite.setPosition(sf::Vector2f(startX, 360.f)); // absolute position
		game_window.draw(piece_sprite);
		startX = startX + 60.1f;
	}
}

void Game::Render() {
	game_window.clear(sf::Color::Black);
	Game::RenderBoard();
	Game::RenderPieces();
	game_window.display();
}

void Game::InitWindow() {
	game_window.create(sf::VideoMode(800, 500), "doola chess");
}

bool Game::IsRunning() {
	return game_window.isOpen();
}


