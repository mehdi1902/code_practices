from node import Node
from state import rna_folding_state


class MSTS():
	def __init__(self, root=None):
		if root:
			self.root = root
		else:
			self.root = Node()