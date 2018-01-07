

###### Machine Learning

- Compile a list of game features.
- Featurize past games
- Add strategy parameters
- Start with seed strategy and train it on featurized past games
- Figure out how to save a log of all games our modified bot will play, similarly to the way it is currently done in the bot, only in our own database
- Update our learned strategy every time the bot plays a game
- Train bot also on IRC data

#### Models

- Use LSTM to classify call and raise decisions as whether they lead to winning or losing
- Learn how to bluff: learn behavior of players that won without showdown, and learn how to
not act like players who reached showdown and had nothing in their hand.
- Classify  players that had good hands and predict how they behave.
- Predict given a player's actions whether they are likely to quit or whether they have a good hand

## Model idea for reinforcement learning from individual IRC players:

X is a matrix in which each row is a vector that represents a game state
y is a vector of numbers where the i-th row represents a good player's move on the i-th game state
f is some non-linear transformation (a neural network) from the space of X to the space of possible strategies

Observation: X
Action: applying strategy f(x) to game state x.
Reward: figure out how to calculate which parts of the strategy took us farther away from y

Problem: only limited to players that showed their cards.

To run gcp job: 

	Locally:
	# From ..
	gcloud ml-engine local train --module-name POD.gcp_task --package-path POD/
	
	Cloudly:	
	# From ..
	gcloud ml-engine jobs submit training peanut_butter_`date +%s` --job-dir=gs://omerkorat2/pb --module-name POD.gcp_task --package-path POD/ --region us-central1

    Stream log:
    gcloud ml-engine jobs stream-logs peanut_butter_1514754990

###### Site Deception

1. Imitate human mouse.
2. Make emotional decisions sometimes.
3. Remember to disable collusion!

