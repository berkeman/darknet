from functools import reduce
from operator import mul
from itertools import chain
from sys import argv

assert len(argv)==2

class Net(object):
	def __init__(self, w, h, c):
		self.v = []
		self.dims = dict(w = int(w), h = int(h), c = int(c),)
		self.input = dict(
			type = 'input',
			dims = self.dims,
		)

	def _abslayer(self, layer):
		return len(self.v) + layer if layer < 0 else layer

	def conv(self, stride, filters, size):
		macs = filters * size * size * self.dims['c'] * \
		       self.dims['w'] * self.dims['h'] \
		       // stride // stride
		self.dims = dict(
			w = self.dims['w'] // stride,
			h = self.dims['h'] // stride,
			c = filters,
		)
		self.v.append(dict(
			type = 'convolve',
			params = "%dx%d/%dx%d" % (size, size, stride, filters),
			dims = self.dims,
			macs = macs,
		))

	def upsample(self, stride):
		for x in 'wh':
			self.dims[x] = stride * self.dims[x]
		self.v.append(dict(
			type = 'upsample',
			params = stride,
			dims = self.dims,
		))

	def shortcut(self, layer):
		# elementwise add <layer> to previous layer
		layer = self._abslayer(layer)
		self.dims = self.v[layer]['dims']
		self.v.append(dict(
			type = 'shortcut',
			params = layer,
			dims = self.dims,
		))

	def maxpool(self, stride, size):
		self.dims = dict(
			w = self.dims['w'] // stride,
			h = self.dims['h'] // stride,
			c = self.dims['c'],
		)
		self.v.append(dict(
			type = 'convolve',
			params = "%dx%d/%dx" % (size, size, stride),
			dims = self.dims,
		))

	def route(self, layers):
		# route will just take info from <layers> and
		# concatenate them (if more than one)
		# without any processing.
		# (Same as "concat" in Caffee, I've been told.)
		layers = [self._abslayer(x) for x in layers]
		if len(layers) == 1:
			self.dims = self.v[layers[0]]['dims']
		elif len(layers) == 2:
			for x in 'wh':
				assert self.v[layers[0]]['dims'][x] == self.v[layers[1]]['dims'][x]
			self.dims['c'] = sum(self.v[l]['dims']['c'] for l in layers)
		else:
			exit(-1)
		self.v.append(dict(
			type = 'route',
			params = ','.join(str(x) for x in layers),
			dims = self.dims,
		))

	def yolo(self):
		self.v.append(dict(
			type = 'yolo',
			dims = {},
		))

	def show(self):
		v = self.v
		if len(v):
			x = v[-1]
		else:
			x = self.input
		size = reduce(mul, (x.get('dims', {}).get(key, 0) for key in 'whc'))
		print("%3d %-12s %-12s %4d %4d %4d %12d %12d" % (
			len(v)-1,
			x['type'],
			x.get('params', '-'),
			x['dims'].get('h', 0),
			x['dims'].get('w', 0),
			x['dims'].get('c', 0),
			size,
			x.get('macs', 0),
		))

N = None
net = {}
context = None
pset = {}
with open(argv[1], 'rt') as fh:
	for line in chain(fh, "\n[done]\n"):
		line = line.rstrip('\n')
		if line.startswith('['):
			if context == '[net]':
				N = Net(pset['width'], pset['height'], pset['channels'])
			elif context == '[convolutional]':
				N.conv(pset['stride'], pset['filters'], pset['size'])
			elif context == '[upsample]':
				N.upsample(pset['stride'])
			elif context == '[maxpool]':
				N.maxpool(pset['stride'], pset['size'])
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
