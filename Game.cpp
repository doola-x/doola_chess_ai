#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "Game.hpp"
#include "Square.hpp"

Game::Game() {
	InitWindow();
	InitGame();
}

Game::~Game(){
	//destroy something
}

void Game::Update(sf::Clock dBounce) {
	//do some updating
	HandleEvents(dBounce);
}

void Game::Render() {
	game_window.clear(sf::Color::Black);
	Game::RenderBoard();
	Game::RenderPieces();
	game_window.display();
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

	                std::cout << "Clicked at Sqaure: (" << Square::getSquareFromClick(mousePos.x, mousePos.y)<< ")" << std::endl;
	                std::cout << "Mouse at: (" << mousePos.x << "," << mousePos.y << ")"<< std::endl;
	            }
	        // mouse released
            case sf::Event::MouseButtonReleased:
                if (game_event.mouseButton.button == sf::Mouse::Left && elapsed.asSeconds() > 0.001) {
                    sf::Vector2i releasePos = sf::Mouse::getPosition(game_window);

                    std::cout << "Mouse Released at Sqaure: (" << Square::getSquareFromClick(releasePos.x, releasePos.y)<< ")" << std::endl;
                    std::cout << "Debounce clock time at: " << elapsed.asSeconds() << std::endl;
                }

	        // dont process other types of events
	        default:
	            break;
	    }
	}
}

void Game::RenderBoard() {
	//do some board rendering
	board_sprite.setTexture(board_texture);
	board_sprite.setPosition(sf::Vector2f(10.f, 10.f)); // absolute position
	game_window.draw(board_sprite);
}


void Game::RenderPieces() {
	//TODO :: create a global allPieces[] array and render from there
		// this would include mapping squares to numerical coordinates somehow. function in Square somewhere?
		// global scale variable -- PIECE_SCALE ? calculated relative to window size
		// this needs to happen for the board also -- BOARD_SCALE
		// and positioning as well...

		// 37.125 for distance between ranks
		//
	for (int i = 0; i < 64; i++){
		//allsquares[i] is our piece to render 
		switch (allSquares[i]->piece->getType()){
			case 'p':
				piece_sprite.setTexture(pawn_texture);
				piece_sprite.setScale(.38f, .38f);
				if (allSquares[i]->piece->allegience == 0){
					//white pawn
					std::array<float, 2> squareToRender = allSquares[i]->getCoordsFromSquare();
					piece_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
				} else {
					//black pawn
					std::array<float, 2> squareToRender = allSquares[i]->getCoordsFromSquare();
					piece_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
				}
				game_window.draw(piece_sprite);

			case 'r':
				rook_sprite.setTexture(rook_texture);
				rook_sprite.setScale(.38f, .38f);

				if (allSquares[i]->piece->allegience == 0){
					//white rook
					std::array<float, 2> squareToRender = allSquares[i]->getCoordsFromSquare();
					rook_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
				} else {
					//black rook
					std::array<float, 2> squareToRender = allSquares[i]->getCoordsFromSquare();
					rook_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
				}
				game_window.draw(rook_sprite);
		}
	}

}

void Game::InitWindow() {
	game_window.create(sf::VideoMode(800, 500), "doola chess");
}

void Game::InitGame() {
	// do some game initializing
	// here we need to set the correct squares equal to the correct piece type
	// lets do correct squares here as well

	char file[8] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
	char rank[8] = {'1', '2', '3', '4', '5', '6', '7', '8'};
	char pieceTypes[8] = {'r', 'n', 'b', 'k', 'q', 'b', 'n', 'r'};

	int count = 0;
	for (int i = 0; i < 8; i++){
		for (int j = 0; j < 8; j++){
			allSquares[count] = new Square(file[j], rank[i]);
			count++;
		}
	}

	for (int i = 0; i < 8; i++){
		allSquares[i]->piece = new Piece(0, pieceTypes[i]);
		allSquares[63-i]->piece = new Piece(1, pieceTypes[i]);
	}

	for (int i = 8; i < 16; i++){
		allSquares[i]->piece = new Piece(0, 'p');
		allSquares[63-i]->piece = new Piece(1, 'p');
	}

	for (int i = 16; i < 48; i++){
		allSquares[i]->piece = new Piece(-1, 'u');
	}

	if (!pawn_texture.loadFromFile("images/tatiana/pw.png"))
	{
	    // error...
	}

	if (!rook_texture.loadFromFile("images/tatiana/rw.png"))
	{
		// error...
	}

	if (!board_texture.loadFromFile("images/board2.png"))
	{
	    // error...
	}
}

bool Game::IsRunning() {
	return game_window.isOpen();
}


