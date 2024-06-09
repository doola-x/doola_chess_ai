# doola chess ai, a policy/value network with GUI developed and trained by me
	- model/network architecture all developed with pytorch and numpy
	- GUI in c++, using SFML (simple fast media layer)
	- some shell scripts to fetch my public game data and util stuff

# policy network
	- the policy network is 3 layers:
		- convolutional layer
		- lstm layer
		- final fully connected layers
			- (i still don't fully understand what is going on here, but i have read alot about RNNs/CNNs and know that we are taking advantage of spatial reasoning as well as some form of memory with LSTM)
	- the network takes in a tensor representation of a current board state 
		- these tensors are created during the data processing phase using the function fen_to_tensor
		- the "correct" move in any given position is the move that i played

# inference
	- inference on the policy network works just like training.
		- the main function takes a fen string, converts it to a tensor
		- this tensor is used as input for the network, and the final layer has all 9010 legal moves mapped to an individual neuron
		- the outputs are decoded, and the legal move with the highest activation score is provided as the suggested move

# value network
	- my current goal revolves around implementing a value network to operate a Sequential Monte Carlo planning mechanism's critic role
	- i plan to manipulate the original 8x8x13 (positionxpiece_type) to 9x8x13 to account for activation scores of different tactical stratagies i enjoy using in my real games
	- this new tensor would allow for a true value to be assigned to the current position based on possible discounted future rewards
		- these rewards will be based on simple W/L as well as the tactics i define