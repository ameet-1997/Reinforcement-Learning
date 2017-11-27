# Checking performance of epsilon greedy method and softmax method on a 10-armed bandit problem

import numpy as np
import matplotlib.pyplot as plt

# Create the 10-arm problem with gaussian rewards
arms = np.random.normal(size=(10))

# Using epsilon greedy method
eps = 0.01
optimal_move = np.random.randint(low=0, high=arms.shape[0])

# Average over 2000 games
# Play for 1000 moves in each game
average_rewards = np.zeros(shape=1000)
number_of_games = 20000

for games in range(number_of_games):
	running_average = 0.0
	for i in range(1000):
		cur_rand = np.random.uniform()
		current_move = optimal_move
		if cur_rand < eps:
			current_move = np.random.randint(low=0, high=arms.shape[0])
		if arms[current_move] > arms[optimal_move]:
			optimal_move = current_move
		running_average = running_average + arms[current_move]
		average_rewards[i] += running_average/(i+1)

average_rewards /= number_of_games

# Plot the graph for an epsilon value
plt.plot([(i+1) for i in range(1000)], list(average_rewards))
plt.ylabel('Average Rewards')
plt.xlabel('Number of moves')
plt.show()
