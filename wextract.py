
from sys import argv

assert len(argv)==2

class Net(object):
	def __init__(self, w, h, c):
		self.v = []
		w, h, c = (int(w), int(h), int(c))
		self.input = (w, h, c)
		self.current = (w, h, c)

	def _abslayer(self, layer):
		return len(self.v) + layer if layer < 0 else layer

	def conv(self, stride, filters):
		self.current = (
			self.current[0]//stride,
			self.current[1]//stride,
			filters,
		)
		self.v.append(['convolve', self.current])

	def upsample(self, stride):
		self.current = (
			self.current[0]*stride,
			self.current[1]*stride,
			self.current[2],
		)
		self.v.append(['upsample', self.current])

	def shortcut(self, layer):
		# plain copy from layer
		layer = self._abslayer(layer)
		assert self.v[layer][1] == self.v[len(self.v)-1][1]
		self.v.append(['upsample', self.current])

	def route(self, layers):
		# route will just take info from <layers> and
		# concatenate them (if more than one)
		# without any processing.
		# (Same as "concat" in Caffee, I've been told.)
		layers = [self._abslayer(x) for x in layers]
		if len(layers) == 1:
			self.current = self.v[layers[0]][1]
		elif len(layers) == 2:
			assert self.v[layers[0]][1][0:2] == self.v[layers[1]][1][0:2]
			self.current = (
				self.current[0],
				self.current[1],
				self.v[layers[0]][1][2] + self.v[layers[1]][1][2]
			)
		else:
			exit(-1)
		self.v.append(['route   ', self.current])

	def yolo(self):
		self.v.append(['yolo    ', ()])

	def show(self):
		if len(self.v):
			print(len(self.v)-1, self.v[-1])
		else:
			print(-1, self.input)

N = None
net = {}
context = None
pset = {}
with open(argv[1], 'rt') as fh:
	for line in fh:
		line = line.rstrip('\n')
		if line.startswith('['):
			if context == '[net]':
				N = Net(pset['width'], pset['height'], pset['channels'])
			elif context == '[convolutional]':
				N.conv(pset['stride'], pset['filters'])
			elif context == '[upsample]':
				N.upsample(pset['stride'])
			elif context == '[shortcut]':
				N.shortcut(pset['from'])
			elif context == '[yolo]':
				N.yolo()
			elif context == '[route]':
				N.route(pset['layers'])
			else:
				if context is not None:
					print('Unknown context', context)
					exit(1)
			context = line
			pset = {}
			if N:
				N.show()
		elif '=' in line:
			param, value = line.replace(' ', '').split('=')
			if param == 'layers':
				value = [int(x) for x in value.split(',')]
			else:
				try:
					value = int(value)
				except Exception:
					pass
			pset[param] = value
