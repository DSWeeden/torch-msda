"""
Follows https://github.com/joshloyal/MarginalizedDenoisingAutoEncoder/blob/master/mda/mda.py
"""

import torch
from torch import Tensor
import numpy as np


def mda(X: Tensor, noise_level: float, W_regularizer: float):

	n_samples, n_features = X.shape
	n_params = n_features + 1

	# Initialize noise vector. Last term is left at one, since bias should not be perturbed.
	q = torch.ones(n_params, 1)
	q[:-1, :] = (1 - noise_level)

	# Initialize scatter matrix
	S = torch.zeros(n_params, n_params)
	S[:n_features, :n_features] = torch.mm(X.T, X)

	# Initialize bias part of scatter matrix
	feature_sum = torch.sum(X, axis=0)
	S[-1, :-1] = feature_sum
	S[:-1, -1] = feature_sum
	S[-1, -1] = n_samples

	# Q matrix (n_feature + 1, n_feature + 1) matrix
	Q = S * torch.mm(q, q.T)
	# torch.fill_diagonal(Q, q * torch.diag(S))
	v = q * torch.diag(S)
	mask = torch.diag(torch.ones_like(v))
	out = mask*torch.diag(v) + (1. - mask)*Q

	# P matrix (n_features, n_features + 1) matrix
	q_tiled = torch.tile(q.T, (n_features,  1))
	P = S[:-1, :] * q_tiled

	# regularization term
	reg = torch.eye(n_params) * W_regularizer
	reg[-1, -1] = 0.

	weights = torch.lstsq(Q + reg, P.T)[0]
	weights = weights[:-1, :]
	biases = weights[-1, :]

	print(weights)
	print(biases)
	return weights, biases


if __name__ == '__main__':
	torch.manual_seed(0)
	x = torch.rand([10, 4], dtype=torch.float32)
	n = 0.4
	w = 0.5
	mda(x, n, w)
