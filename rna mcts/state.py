import numpy as np
from RNA import energy_of_structure
MIN_GAP = 4

class RNAFoldingState():
	def self.__init__(self, structure):
		"""
		Args:
			structure: secondary or tertiary structure 
					e.g.: [10, 9, 8, -1, -1, -1, -1, -1, 3, 2, 1]
		"""
		self.structure = structure
		self.N = len(structure)

	def is_leaf(self):
		""" 
		Check if the state is a leaf one or not
		** knowing a leaf node is tricky but as a simple method
		we consider a structure without any unpaired nucleotide as leaf
		It means you cannot go furthur. It's obvious that solution with 
		MFE can be a structure which is not a leaf :)
		"""
		return not -1 in self.structure