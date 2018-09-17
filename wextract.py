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

	def conv(self, stride, filters, size, padding=0):
		if padding:
			# assume "same"
			pad = (size - 1) // 2
		else:
			pad = 0
		w = ((self.dims['w'] + 2*pad - size) // stride) + 1
		h = ((self.dims['h'] + 2*pad - size) // stride) + 1
		weights = filters * (size * size * self.dims['c'] + 1)
		macs = (size * size * self.dims['c']) * (w * h * filters)
		self.dims = dict(w=w, h=h, c=filters,)
		self.v.append(dict(
			type = 'convolve',
			params = "%dx%d/%dx%d%s" % (size, size, stride, filters, 'p' if padding else ''),
			dims = self.dims,
			macs = macs,
			weights = weights,
		))

	def local(self, stride, filters, size, padding=0):
		if stride == 1:
			# assume "same"
			pad = (size - 1) // 2
		else:
			pad = 0 # assumption going on here!
		w = ((self.dims['w'] + 2*pad - size) // stride) + 1
		h = ((self.dims['h'] + 2*pad - size) // stride) + 1
		weights = filters * (size * size * self.dims['c'] + 1) * w * h
		macs = (size * size * self.dims['c']) * (w * h * filters)
		self.dims = dict(w=w, h=h, c=filters,)
		self.v.append(dict(
			type = 'local',
			params = "%dx%d/%dx%d%s" % (size, size, stride, filters, 'p' if padding else ''),
			dims = self.dims,
			macs = macs,
			weights = weights,
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

	def maxpool(self, stride, size, padding):
		if stride == 1:
			# assume "same"
			pad = (size - 1) // 2
		else:
			pad = padding
		self.dims = dict(
			w = ((self.dims['w'] + 2*pad - size) // stride) + 1,
			h = ((self.dims['h'] + 2*pad - size) // stride) + 1,
			c = self.dims['c'],
		)
		self.v.append(dict(
			type = 'maxpool',
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

	def detection(self):
		self.v.append(dict(
			type = 'detection',
			dims = {},
		))

	def connected(self, output):
		self.dims = dict(w = output, h=1, c=1)
		x = self.v[-1]['dims']
		macs = output * x['w'] * x['h'] * x['c']
		self.v.append(dict(
			type = 'fc',
			macs = macs,
			dims = self.dims,
			weights = output * (x['w'] * x['h'] * x['c'] + 1)
		))

	def dropout(self):
		self.v.append(dict(
			type = 'dropout',
			dims = self.dims,
		))

	def softmax(self):
		self.v.append(dict(
			type = 'softmax',
			dims = self.dims,
		))


	def show(self):
		v = self.v
		if len(v):
			x = v[-1]
		else:
			x = self.input
		size = reduce(mul, (x.get('dims', {}).get(key, 0) for key in 'whc'))
		print("%3d %-12s %-12s %4d %4d %4d %12d %12d %12d" % (
			len(v)-1,
			x['type'],
			x.get('params', '-'),
			x['dims'].get('h', 0),
			x['dims'].get('w', 0),
			x['dims'].get('c', 0),
			size,
			x.get('macs', 0),
			x.get('weights', 0),
		))


print("#   layer                        w    h    c         feat         macs      weights")
print("#----------------------------------------------------------------------------------")
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
				N.conv(pset['stride'], pset['filters'], pset['size'], pset['pad'])
			elif context == '[local]':
				N.local(pset['stride'], pset['filters'], pset['size'], pset['pad'])
			elif context == '[upsample]':
				N.upsample(pset['stride'])
			elif context == '[maxpool]':
				N.maxpool(pset['stride'], pset['size'], pset.get('padding', 0))
			elif context == '[shortcut]':
				N.shortcut(pset['from'])
			elif context == '[connected]':
				N.connected(pset['output'])
			elif context == '[yolo]':
				N.yolo()
			elif context == '[detection]':
				N.detection()
			elif context == '[route]':
				N.route(pset['layers'])
			elif context == '[dropout]':
				N.dropout()
			elif context == '[softmax]':
				N.softmax()
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
