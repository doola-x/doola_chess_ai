#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <functional>
#include "Game.hpp"
#include "Square.hpp"

Game::Game() {
	InitWindow();
	InitGame();
	InitSocket();
}

Game::~Game(){
	//destroy something
    isRunning = false;
    cv.notify_all();
    if (commThread.joinable()) {
        commThread.join();
    }
    close(sock);
    for (int i = 0; i < 64; i++) {
        delete allSquares[i]->piece;
        delete allSquares[i];
    }
}

void Game::InitSocket() {
    sock = 0;
    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cout << "Socket creation error" << std::endl;
        return;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(65432);

    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        std::cout << "Invalid address/ Address not supported" << std::endl;
        return;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cout << "Connection Failed" << std::endl;
        return;
    }

    isRunning = true;
    commThread = std::thread([this]() { this->CommWithPython(); });
}


void Game::CommWithPython() {
    while (isRunning) {
        std::string move;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this]() { return !moveQueue.empty() || !isRunning; });
            if (!isRunning) break;
            move = moveQueue.front();
            moveQueue.pop();
        }

        // Send move to Python
        send(sock, move.c_str(), move.length(), 0);

        // Receive response from Python
        char buffer[1024] = {0};
        int valread = read(sock, buffer, 1024);
        {
            std::lock_guard<std::mutex> lock(mtx);
            pythonResponse = std::string(buffer, valread);
        }
    }
}

int Game::SendMoveToPython(const std::string& move) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        moveQueue.push(move);
    }
    cv.notify_one();
    return 0;
}

void Game::ProcessPythonResponse(const std::string& response) {
	if (response == "O-O" || response == "O-O-O"){
		std::cout << response << std::endl;
		return;
	}
    char file = response[0];
    int rank = response[1] - '0';
    int fileMult = file - 'a';
    int rankMult = 8 * (rank - 1);
    selectedSquare = rankMult + fileMult;
    std::cout << "Python selected square: " << selectedSquare << std::endl;

    file = response[2];
    rank = response[3] - '0';
    fileMult = file - 'a';
    rankMult = 8 * (rank - 1);
    int dropSquare = rankMult + fileMult;

    if (dropSquare > -1 && dropSquare < 64) {
        if (selectedSquare > -1 && selectedSquare < 64) {
            DoMove(dropSquare, false);
            isUserTurn = true;
        }
    } else {
        std::cout << "Illegal drop square!" << std::endl;
    }
}

void Game::Update(sf::Clock dBounce) {
	HandleEvents(dBounce);
	std::string response;
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!pythonResponse.empty()) {
            response = pythonResponse;
            pythonResponse.clear();
        }
    }
    if (!response.empty()) {
        ProcessPythonResponse(response);
    }
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
	        	move_str = "";
	            if (game_event.mouseButton.button == sf::Mouse::Left) {

	                sf::Vector2i mousePos = sf::Mouse::getPosition(game_window);
	                elapsed = dBounce.restart();

	                sqr_c = Square::getSquareFromClick(mousePos.x, mousePos.y);
	                char file = sqr_c[0];
	                int rank = sqr_c[1] - '0';
	                int fileMult = file - 'a';
	                int rankMult = 8 * (rank - 1);

	                selectedSquare = rankMult+fileMult;

	                std::cout << "Clicked at Sqaure: (" << sqr_c << ")" << std::endl;
	                std::cout << "Mouse at: (" << mousePos.x << "," << mousePos.y << ")"<< std::endl;
	            }
	        // mouse released
            case sf::Event::MouseButtonReleased:
                if (game_event.mouseButton.button == sf::Mouse::Left && elapsed.asSeconds() > 0.001) {
                    sf::Vector2i releasePos = sf::Mouse::getPosition(game_window);

                    sqr_r = Square::getSquareFromClick(releasePos.x, releasePos.y);
                    char file = sqr_r[0];
                   	int rank = sqr_r[1] - '0';
	                int fileMult = file - 'a';
	                int rankMult = 8 * (rank - 1);
	                int dropSquare = rankMult + fileMult;
	                if (dropSquare == selectedSquare) {
	                	std::cout << "illegal move!" << std::endl;
	                	break;
	                }
	                if (!isUserTurn) {
	                	std::cout << "be patient!" << std::endl;
	                	break;
	                }
                    if (dropSquare > -1 && dropSquare < 64){
		                if (selectedSquare > -1 && selectedSquare < 64){
							DoMove(dropSquare, true);
		                }
	                    std::cout << "Mouse Released at Sqaure: (" << sqr_r << ")" << std::endl;
	                    std::cout << "Debounce clock time at: " << elapsed.asSeconds() << std::endl;
                    } else {
                    	std::cout << "Illegal drop square!" << std::endl;
                    	move_str = "";
                    }

                }

	        // dont process other types of events
	        default:
	            break;
	    }
	}
}

void Game::DoMove(int dropSquare, bool send) {
	// move game logic here
	char t = allSquares[selectedSquare]->piece->getType();
	int a = allSquares[selectedSquare]->piece->allegience;
	char takes = allSquares[dropSquare]->piece->type;
	char from_col = allSquares[selectedSquare]->file;

	if (send) {
		int valid = rules->isValidMove(t, dropSquare, selectedSquare, takes, allSquares);

		if (valid < 0) {
			std::cout << "invalid move" << std::endl;
			return;
		} else if (valid > 0) {
			std::cout << "valid move" << std::endl;
		}
	}

	move_str = "";
	if (t == 'p'){
		if (takes != 'u') {
			move_str += sqr_c[0];
		}
	} else {
		move_str += std::toupper(t);
	}
	if (t == 'r' || t == 'n') {
		move_str += from_col;
	}
	if (takes != 'u') move_str += 'x';
	move_str += sqr_r;
	// check if move is castling
	if (t == 'k'){
		int spaces = (dropSquare - selectedSquare);
		if (spaces == 2){
			move_str = "O-O";
		}
		else if (spaces == -2){
			std::cout << "move is long castles" << std::endl;
			move_str = "O-O-O";
		}
	}
	std::cout << "Processed move to pass: " << move_str << std::endl;
	allSquares[selectedSquare]->piece->type = 'u';
	allSquares[selectedSquare]->piece->allegience = -1;

	allSquares[dropSquare]->piece->type = t;
	allSquares[dropSquare]->piece->allegience = a;

	if (move_str == "O-O"){
		int rookSquare = -1;
		allSquares[dropSquare - rookSquare]->piece->type = 'u';
		allSquares[dropSquare - rookSquare]->piece->allegience = -1;

		allSquares[dropSquare + rookSquare]->piece->type = 'r';
		allSquares[dropSquare + rookSquare]->piece->allegience = a;
	}
	if (move_str == "O-O-O"){

		int rookSquare = -1;
		allSquares[dropSquare + (rookSquare * 2)]->piece->type = 'u';
		allSquares[dropSquare + (rookSquare * 2)]->piece->allegience = -1;

		allSquares[dropSquare - rookSquare]->piece->type = 'r';
		allSquares[dropSquare - rookSquare]->piece->allegience = a;
	}

	if (send) {
		SendMoveToPython(move_str);
		isUserTurn = false;
	}
}

void Game::RenderBoard() {
	//do some board rendering
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
		if (allSquares[i]->piece != nullptr){
			std::array<float, 2> squareToRender = allSquares[i]->getCoordsFromSquare();
			if (allSquares[i]->piece->getType() == 'p'){
				if (allSquares[i]->piece->allegience == 0){
					//white pawn
					pawn_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
					game_window.draw(pawn_sprite);
				} else {
					//black pawn
					b_pawn_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
					game_window.draw(b_pawn_sprite);
				}
			}
				
			if (allSquares[i]->piece->getType() == 'r'){				
				if (allSquares[i]->piece->allegience == 0){
					//white rook
					rook_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
					game_window.draw(rook_sprite);
				} else {
					//black rook
					b_rook_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
					game_window.draw(b_rook_sprite);
				}
			}

			if (allSquares[i]->piece->getType() == 'n'){				
				if (allSquares[i]->piece->allegience == 0){
					//white rook
					knight_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
					game_window.draw(knight_sprite);
				} else {
					//black rook
					b_knight_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
					game_window.draw(b_knight_sprite);
				}
			}

			if (allSquares[i]->piece->getType() == 'b'){				
				if (allSquares[i]->piece->allegience == 0){
					//white rook
					bishop_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
					game_window.draw(bishop_sprite);
				} else {
					//black rook
					b_bishop_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
					game_window.draw(b_bishop_sprite);
				}
			}

			if (allSquares[i]->piece->getType() == 'k'){				
				if (allSquares[i]->piece->allegience == 0){
					//white rook
					king_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
					game_window.draw(king_sprite);
				} else {
					//black rook
					b_king_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
					game_window.draw(b_king_sprite);
				}
			}

			if (allSquares[i]->piece->getType() == 'q'){				
				if (allSquares[i]->piece->allegience == 0){
					//white rook
					queen_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
					game_window.draw(queen_sprite);
				} else {
					//black rook
					b_queen_sprite.setPosition(sf::Vector2f(squareToRender[0], squareToRender[1])); // absolute position
					game_window.draw(b_queen_sprite);
				}
			}
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
		allSquares[i]->piece = new Piece(0, pieceTypes[7-i]);
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
	pawn_sprite.setTexture(pawn_texture);
	pawn_sprite.setScale(.38f, .38f);

	if (!b_pawn_texture.loadFromFile("images/tatiana/pb.png")){
		// error...
	}
	b_pawn_sprite.setTexture(b_pawn_texture);
	b_pawn_sprite.setScale(.38f, .38f);

	if (!rook_texture.loadFromFile("images/tatiana/rw.png"))
	{
		// error...
	}
	rook_sprite.setTexture(rook_texture);
	rook_sprite.setScale(.38f, .38f);

	if (!b_rook_texture.loadFromFile("images/tatiana/rb.png"))
	{
		// error...
	}
	b_rook_sprite.setTexture(b_rook_texture);
	b_rook_sprite.setScale(.38f, .38f);

	if (!knight_texture.loadFromFile("images/tatiana/nw.png"))
	{
		// error...
	}
	knight_sprite.setTexture(knight_texture);
	knight_sprite.setScale(.38f, .38f);

	if (!b_knight_texture.loadFromFile("images/tatiana/nb.png"))
	{
		// error...
	}
	b_knight_sprite.setTexture(b_knight_texture);
	b_knight_sprite.setScale(.38f, .38f);

	if (!bishop_texture.loadFromFile("images/tatiana/bw.png"))
	{
		// error...
	}
	bishop_sprite.setTexture(bishop_texture);
	bishop_sprite.setScale(.38f, .38f);

	if (!b_bishop_texture.loadFromFile("images/tatiana/bb.png"))
	{
		// error...
	}
	b_bishop_sprite.setTexture(b_bishop_texture);
	b_bishop_sprite.setScale(.38f, .38f);

	if (!king_texture.loadFromFile("images/tatiana/kw.png"))
	{
		// error...
	}
	king_sprite.setTexture(king_texture);
	king_sprite.setScale(.38f, .38f);

	if (!b_king_texture.loadFromFile("images/tatiana/kb.png"))
	{
		// error...
	}
	b_king_sprite.setTexture(b_king_texture);
	b_king_sprite.setScale(.38f, .38f);

	if (!queen_texture.loadFromFile("images/tatiana/qw.png"))
	{
		// error...
	}
	queen_sprite.setTexture(queen_texture);
	queen_sprite.setScale(.38f, .38f);

	if (!b_queen_texture.loadFromFile("images/tatiana/qb.png"))
	{
		// error...
	}
	b_queen_sprite.setTexture(b_queen_texture);
	b_queen_sprite.setScale(.38f, .38f);

	if (!board_texture.loadFromFile("images/board2.png"))
	{
	    // error...
	}
	board_sprite.setTexture(board_texture);
	board_sprite.setPosition(sf::Vector2f(10.f, 10.f)); // absolute position

	rules = new Rules();
	isUserTurn = true;
}

bool Game::IsRunning() {
	return game_window.isOpen();
}


