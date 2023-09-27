# build plan
  
## goals
  * build a gui with SFML (simple and fast media layer) to communicate with the stockfish engine
  * allow this gui/application to be embedded in my website so visitors can play games
  * eventually plan to customize and train stockfish to play like me (poorly), and from there begin to create my own chess engine with an orginal neural network

## build a gui to communicate with stockfish

this will be one of my first in-depth gui projects with c++. i have a basic level of familiarity with the language but it will be a learning process. i plan to use SFML to handle window creation and graphic rendering, gcc to compile the code, along with assitance from chat gpt to fill the gaps in my understanding. 

the first thing to do (in my misguided opinion) is to create the window where the game will live. i should start with a UserInterface class that can handle functions like windowOpen(), windowUpdate(), and windowRender(). these methods will all need to be passed an instance of another class i need to create, chessboard.

what i can do for now is create the UserInterface class and the Chessboard class. the chessboard class will be bare bones while i get SFML implemented, and once i have the correct assets i will begin work on the chessboard class.

## updated ideas and goals, sept. 17 2023

working with SFML has been an instructive (and difficult) challenge but i have some basic event handling going along with a window and a sprite. new plan is to figure out how to construct a chessboard style clickable-grid. i also want to seperate window startup functionality into another file.. (window_startup.cpp?)

## updated thots and planz sept. 26 2023

i have the grid clickable, although it is a fixed grid and if the window is manipulated at all it gets out of wack. i've began defining classes for square and piece, and the goal for the next week or so is to define sub-classes of piece for all the different types (pawn, rook, bishop, knight, king, queen). beginning to remember about header files and c++ development rules (famous last words). will continue on for now .... (jus bild it)