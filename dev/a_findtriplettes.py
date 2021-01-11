from collections import namedtuple

datasets = ('config',)

columns = ('ci','sx','co','wo','layer','wi','n','mac','stride','loepnummer','hi','ho','groups','sy')
config = namedtuple('config', columns)


def synthesis(params):
	v = tuple(x for x in map(lambda x: config(*x), datasets.config.iterate("roundrobin", columns)))

	unused_layers = set(x.loepnummer for x in v)

	print('Bottlenet combos, not counting stride==2 ones:')
	n = 0
	triplettes = {}
	for ix in range(2, len(v)):
		d2, d1, d0 = v[ix - 2:ix + 1]
		print('#  ', ix, d2.loepnummer, d1.loepnummer, d0.loepnummer)
		if d2.sx == 1 and \
		   d1.sx == 3 and (d1.stride == 1 or d1.stride == 2) and d1.groups == d1.ci == d1.co and\
		   d0.sx == 1:
			triplettes[n] = (d2,d1,d0)
			unused_layers.remove(d2.loepnummer)
			unused_layers.remove(d1.loepnummer)
			unused_layers.remove(d0.loepnummer)
			n += 1
			print(n, d2, d1, d0)
	return triplettes, unused_layers
