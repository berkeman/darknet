from collections import namedtuple

from . import darknetlayer
from . import memory
from . import convlayer

depend_extra = (memory, convlayer, darknetlayer)

jobids  = ('darknet',) # directory with inputs/weights/outputs, one file per layer
datasets= ('config',)  # dataset with network configuration

def synthesis():
	columns = ('ci','sx','co','wo','layer','wi','n','mac','stride','loepnummer','hi','ho','groups','sy')
	config = namedtuple('config', columns)
	v = tuple(x for x in map(lambda x: config(*x), datasets.config.iterate(None, columns)))

	print('Bottlenet combos, not counting stride==2 ones:')
	n = 0
	for ix in range(2, len(v)):
		d2, d1, d0 = v[ix-2:ix+1]
		if d2.sx == 1 and \
           d1.sx == 3 and d1.stride == 1 and d1.groups == d1.ci == d1.co and\
           d0.sx == 1:
			print()
			print('%2d  ' %(n,) + str(d2))
			print('%2d  ' %(n,) + str(d1))
			print('%2d  ' %(n,) + str(d0))
			n += 1
