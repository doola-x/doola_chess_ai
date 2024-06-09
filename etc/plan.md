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

## updated thoughts and plans sept. 26 2023

i have the grid clickable, although it is a fixed grid and if the window is manipulated at all it gets out of wack. i've began defining classes for square and piece, and the goal for the next week or so is to define sub-classes of piece for all the different types (pawn, rook, bishop, knight, king, queen). beginning to remember about header files and c++ development rules (famous last words). will continue on for now .... (jus bild it)

##update oct. 27 2023

started seperating game functionality into its own class. things are making more sense each time i can find time to work on this (oof). i think the next logical step would be to get all the pieces to render, and **then** start building the piece class files. also need some game logic, like creating all the square and piece objects, game rules, etc.

##update feb. 12 2024

oh ya, the model fucking works btw :O watching it train and do inference is the most satisfying thing on the planet. the rules engine is still extremely early stages, and i need a way to determine legal moves as well. i think the model would benefit greatly from this as well.

we did it joe

## update april 28 2024

oh ya, the model fucking sucks btw :-O getting it to work was a really cool process and i learned a shit ton of exteremly useful things along the way. i started initially with just messing around with the data, realized i needed to go back and time and did mnist top 10 to see an actual nn work. from there, i read a lot and decided i just need to go for it. that was in between oct-feb, and data processing from chess.com and structuring positions -> tensors ended up being the bulk of the work. bad data still plagues me... i digress.

what started off as a fully feed forward network evolved into custom loss functions with scalar penlties for illegal suggestions, enhanced inference protocols, updated game/tactics content and what is now looking like a reinforcement learning setup with sequential monte carlo planning for expert iteration. i am currently implementing this paper (kind of): https://arxiv.org/pdf/2402.07963

tonight im cleaning up the cpp game code a little bit (lots of work to do here) and implementing socket handling for my inference script and the c++ executable to communicate. im re-building my professional site with sveltekit and hope to host the model in an environment that is actually usable before ? (summer? fall? how much work do i even have to do?)

i can also implement a chess engine for c++ just for bragging rights, but thinking a quick fix for now is pass the move over through a socket to the play_inference python script and do a quick check for legality in the current position, and just send a bit flag over to the executable and allow play to continue. if the move is illegal, we bail 