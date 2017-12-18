

Model for reinforcement learning from individual players:

X is a matrix in which each row is a vector that represents a game state
y is a vector of numbers where the i-th row represents a good player's move on the i-th game state
f is some non-linear transformation (a neural network) from the space of X to the space of possible strategies

The observation is X, the action is the result of applying strategy f(x) to game state x, and the reward is the cross entroy of f(X) and y
