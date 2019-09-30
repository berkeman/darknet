from dataset import DatasetWriter

from collections import namedtuple

datasets=('source',)

columns = ('layer', 'sx', 'sy', 'groups', 'stride', 'wi', 'hi', 'ci', 'wo', 'ho', 'co',)

Layer = namedtuple('Layer', columns)

def window(v, N=3):
	# return tuple (x_n, x_{n+1}, x_{n+N-1}) for each n.
	# NB: Does not return head or tail.
	buf = []
	for n, item in enumerate(v):
		buf.append(item)
		buf = buf[-N:]
		if len(buf)==N:
			yield (n, buf)


def analysis(sliceno):

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
	for data in datasets.source.iterate(sliceno, columns):
		v.append(Layer(*data))


	pops = set()
	for n, (l0, l1, l2) in window(v, 3):
		if (l0.ci <= l1.ci >= l2.co) and (l0.sx==l0.sy==l2.sx==l2.sy==1) and (l1.sx==l1.sy==3) and (l1.groups==l1.ci==l1.co):
			dw.write(l0.wi, l0.hi, l2.wo, l2.ho, l0.ci, l0.co, l2.co, l1.stride)
			pops.update({n-2, n-1, n})
	dw.finish()


	dw = DatasetWriter(name='remlayers')
	for key in columns:
		dw.add(datasets.source.columns[key].name, datasets.source.columns[key].type)

	for n, x in enumerate(v):
		if n in pops:
			continue
		dw.write(*x)
	dw.finish()
