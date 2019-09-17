class Layer():
	def __init__(self, filename):
		self.filename = filename
		self.fh = open(filename, 'rt')
		self.nextlayer()
	def nextlayer(self):
		line = self.fh.readline().rstrip('\n').split(' ')
		self.hi, self.wi, self.ci, self.groups, self.k, self.co, self.stride = map(int, line[1:])
		line = self.fh.readline().rstrip('\n').split(' ')
		self.ho, self.wo, _, self.pad = map(int, line[1:])
		line = self.fh.readline().rstrip('\n').split(' ')
		_, _, self.bn, self.activation = map(int, line[1:])
		self.activation = {1:'relu', 3:'linear'}[self.activation]
		def getdata():
			x = self.fh.readline()
			return tuple(map(float, x.rstrip().split(' ')))
		self.weights = getdata()
		self.biases  = getdata()
		self.inputs  = getdata()
		if self.bn:
			self.rolling_mean = getdata()
			self.rolling_var = getdata()
			self.scales = getdata()
		self.outputs = getdata()   # scalar prod output
		self.outputs2 = getdata()  # with norm + bias
		self.outputs3 = getdata()  # with activations
		self.outputs = [0 for x in range(len(self.outputs))] # kill this to ensure that normalisation and bias is active!
		self.outputs2 = [0 for x in range(len(self.outputs))] # kill this to ensure that normalisation and bias is active!
