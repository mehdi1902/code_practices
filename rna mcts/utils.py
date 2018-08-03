import numpu as np
from math import sqrt, ln


def uct(node):
	""" Calculate the uct values for a node and its children
	Args:
		node: an object from the node class with "visits" and "children list"
	"""
	weights = []
	total_weight = 0.
	for child in node.children:
		w = child.value + C * sqrt(ln(node.visits) / child.visits)
		total_weight += w
		weights.append(w)
	return np.array(weights) / total_weight