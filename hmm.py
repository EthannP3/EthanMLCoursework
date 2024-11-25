import numpy as np
from hmmlearn import hmm
# import matplotlib.pyplot as plt
# from sympy.stats import TransitionMatrixOf
# NOTE: Use hmm environment instead of standard

rewards = np.loadtxt('rewards.txt', dtype=int)
NStates = 9
NObs = 3
# TransitionMatrix
ModelNoTransition = hmm.CategoricalHMM(n_components=NStates, n_iter=100, tol=1e-4, random_state=42)
ModelNoTransition.n_features = NObs
print(type(rewards))
ShapedRewards = rewards.reshape(-1, 1)
print(ShapedRewards.shape)
ModelNoTransition.fit(ShapedRewards)
# Learned parameters
learned_start_prob = ModelNoTransition.startprob_
learned_trans_matrix = ModelNoTransition.transmat_
learned_emission_matrix = ModelNoTransition.emissionprob_

print("Learned Start Probabilities (without transitions):")
print(learned_start_prob)
print("Learned Transition Matrix (without transitions):")
print(learned_trans_matrix)
print("Learned Emission Matrix (without transitions):")
print(learned_emission_matrix)

grid_size = 3
n_states = grid_size * grid_size
true_transition_matrix = np.zeros((n_states, n_states))

def get_neighbors(state, grid_size):
    x, y = divmod(state, grid_size)
    neighbors = []
    if x > 0: neighbors.append((x - 1) * grid_size + y)
    if x < grid_size - 1: neighbors.append((x + 1) * grid_size + y)
    if y > 0: neighbors.append(x * grid_size + (y - 1))
    if y < grid_size - 1: neighbors.append(x * grid_size + (y + 1))
    return neighbors
for from_state in range(n_states):
    neighbors = get_neighbors(from_state, grid_size)
    for to_state in neighbors:
        true_transition_matrix[from_state, to_state] = 1 / len(neighbors)

print("Transition Matrix:")
print(true_transition_matrix)

assert np.allclose(true_transition_matrix.sum(axis=1), 1)
ModelTransition = hmm.CategoricalHMM(n_components=NStates, n_iter=100, tol=1e-4, random_state=42)
ModelTransition.n_features = NObs
ModelTransition.transmat_ = true_transition_matrix
ModelTransition.params = 'se'  # Learn emission probabilities only
ModelTransition.init_params = 'se'
ModelTransition.fit(ShapedRewards)
learned_start_prob = ModelTransition.startprob_
learned_emission_matrix = ModelTransition.emissionprob_

print("Learned Start Probabilities (without transitions):")
print(learned_start_prob)
print("Learned Emission Matrix (without transitions):")
print(learned_emission_matrix)