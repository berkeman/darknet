from dataset import DatasetWriter

from collections import namedtuple

datasets=('source',)

Layer = namedtuple('Layer', 'layer sx sy groups stride wi hi ci wo ho co')
ResLayer = namedtuple('ResLayer', 'wi hi wo ho ci cm co stride')

def window(v, N=3):
	# return tuple (x_n, x_{n+1}, x_{n+N-1}) for each n.
	# NB: Does not return head or tail.
	buf = []
	for n, item in enumerate(v):
		buf.append(item)
		buf = buf[-N:]
		if len(buf)==N:
			yield (n, buf)


def synthesis():

	dw = DatasetWriter(name='reslayers')
	dw.add('wi', 'number')
	dw.add('hi', 'number')
	dw.add('wo', 'number')
	dw.add('ho', 'number')
	dw.add('ci', 'number')
	dw.add('cm', 'number')
	dw.add('co', 'number')
	dw.add('stride', 'number')
	dw.set_slice(0)

	v = []
	for data in datasets.source.iterate(None, ('layer', 'sx', 'sy', 'groups', 'stride', 'wi', 'hi', 'ci', 'wo', 'ho', 'co',)):
		v.append(Layer(*data))

	pops = set()
	for n, (l0, l1, l2) in window(v, 3):
		if (l0.ci <= l1.ci >= l2.co) and (l0.sx==l0.sy==l2.sx==l2.sy==1) and (l1.sx==l1.sy==3) and (l1.groups==l1.ci==l1.co):
			dw.write(l0.wi, l0.hi, l2.wo, l2.ho, l0.ci, l0.co, l2.co, l1.stride)
			pops.update({n-2, n-1, n})
	dw.finish()

	remlayers = [x for n, x in enumerate(v) if n not in pops]
	dw = DatasetWriter(name='remlayers')
	dw.add('layer', 'unicode')
	dw.add('sx', 'number')
	dw.add('sy', 'number')
	dw.add('groups', 'number')
	dw.add('stride', 'number')
	dw.add('wi', 'number')
	dw.add('hi', 'number')
	dw.add('ci', 'number')
	dw.add('wo', 'number')
	dw.add('ho', 'number')
	dw.add('co', 'number')
	dw.set_slice(0)

	for x in remlayers:
		dw.write(x.layer, x.sx, x.sy, x.groups, x.stride, x.wi, x.hi, x.ci, x.wo, x.ho, x.co)
	dw.finish()
