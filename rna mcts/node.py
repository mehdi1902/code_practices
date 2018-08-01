class Node():
	"""
	General class for nodes of tree
	"""
	def __init__(self, state, parent=None, visits=0, reward=0, children=[]):
		"""
		Args:
			state:	Representation of each state. e.g. in the Othello is the board
					contains white or black stones. in RNA folding is the current secondary
					or tertiary structure in any shape like a 2d matrix or arc diagram or ...
			parent: Parent of the node in the tree structure, default value is None
					for root or any separated node
			visits: Number of visits, 0 for default value
			reward: Initial reward value. can be number of wins in the Othello or 
					free energy in the rna folding
			children: A list of node's children with an empty list as default
					* Using the children argument can be useful due to create a new parent
					for some children
		"""
		self.state = state
		self.parent = parent
		self.visits = visits
		self.reward = reward
		self.children = children

	def add_child(self, state, visits=0, reward=0, children=[]):
		""" Adding a node to the children list

		Args:
			state: State of new child
			visits: Number of visits
			reward: Reward value
			children: children list
		"""
		child = Node(state, self, visits, reward, children)
		self.children.append(child)

	def update(self, reward):
		""" Update the current node with a new reward

		Args:
			reward: reward value :)
		"""
		self.visits += 1
		self.reward += reward

	def __str__(self):
		""" 
		Overload the str function to print this node
		"""
		text = """#children: %i\n
		visits: %i\n
		reward: %f""" % (len(self.children), self.visits, self.reward)
		return text

	def is_leaf(self):
		"""
		Check if the node is leaf (unexpanded yet)!
		"""
		return len(self.children) == 0