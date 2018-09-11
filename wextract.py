
from sys import argv

assert len(argv)==2

v = {}
layer = 0
nw, nh, nc = None, None, None
net = {}
context = None
pset = {}
with open(argv[1], 'rt') as fh:
	for line in fh:
		line = line.rstrip('\n')
		if line.startswith('['):
			# print previous
			if context == '[net]':
				for x in ['width', 'height', 'channels']:
					net[x] = pset[x]
				nw = net['width']
				nh = net['height']
				nc = net['channels']

			elif context == '[convolutional]':
				new_w = nw // pset['stride']
				new_h = nh // pset['stride']
				new_c = pset['filters']
				print(
					layer,
					'conv',
					nw, nh, nc,
					'|',
					new_w, new_h, new_c,
					nw*nh*nc, new_w*new_h*new_c,
				)
				nw = new_w
				nh = new_h
				nc = new_c
				v[layer] = (new_w, new_h, new_c)
				layer += 1

			elif context == '[upsample]':
				new_w = nw * pset['stride']
				new_h = nh * pset['stride']
				new_c = nc
				print(
					layer,
					'upsample',
					nw, nh, nc,
					'|',
					new_w, new_h, new_c,
					nw*nh*nc, new_w*new_h*new_c,
				)
				nw = new_w
				nh = new_h
				nc = new_c
				v[layer] = (new_w, new_h, new_c)
				layer += 1

			elif context == '[shortcut]':
				fromlayer = int(pset['from'])
				fromlayer = fromlayer if fromlayer>0 else layer+fromlayer
				new_w, new_h, new_c = v[fromlayer]

				print(layer, 'shortcut', fromlayer,
					new_w, new_h, new_c,
					nw*nh*nc, new_w*new_h*new_c,
				)
				v[layer] = (new_w, new_h, new_c)
				layer += 1

			elif context == '[yolo]':
				print(layer, 'yolo')
				layer += 1

			elif context == '[route]':
				# route will just take info from <layers> and
				# concatenate them (if more than one)
				# without any processing.
				# (Same as "concat" in Caffee, I've been told.)
				print('X', pset['layers'])
				layers = pset['layers']
				layers = [layer + x if x<0 else x for x in layers]
				if len(layers) == 1:
					new_w, new_h, new_c = v[layers[0]]
				else:
					new_w, new_h, new_c = v[layers[1]]
					new_c += v[layers[0]][2]

				print(layer, 'route', layers,
					new_w, new_h, new_c,
					nw*nh*nc, new_w*new_h*new_c,
				)
				v[layer] = (new_w, new_h, new_c)
				layer += 1
			else:
				if context is not None:
					print('Unknown context', context)
					exit(1)


			# restart
			context = line
			pset = {}

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
