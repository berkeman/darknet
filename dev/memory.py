from math import ceil

class Memory():
	def __init__(self, naddr, nwords, verbose=False):
		print('[Memory create: %d x %d]' % (naddr, nwords,))
		self.m = [[0 for x in range(nwords)] for y in range(naddr)]
		self.naddr = naddr
		self.nwords = nwords
		self.verbose = verbose
		if self.verbose:
			self.printsize()

	def write(self, a, d):
		assert len(d) == self.nwords
		assert 0 <= a <= self.naddr
		self.m[a] = d
#		print('write %5d' % (a,), ''.join("% 5.2f" % x for x in d))

	def read(self, a):
		assert 0 <= a <= self.naddr
		return self.m[a]

	def printsize(self):
		print('Memory naddr: ', self.naddr)
		print('Memory nwords:', self.nwords)

	def importvec(self, data, width, height, channels):
		vec2mem(self, data, width, height, channels)

	def export(self, width, height, channels):
		return mem2vec(self, width, height, channels)


def vec2mem(mem, data, width, height, channels):
	"""
	Store "data" in "mem" as a cube "width"x"height"x"channels".
	There are "mem.nwords" channels per address.
	Width first, then height, then channel
	"""
	assert width * height * channels // mem.nwords <= mem.naddr
	depth = ceil(channels / mem.nwords)
	for y in range(height):
		for x in range(width):
			for chigh in range(depth):
				dstadr = x + y * width + chigh * width * height
				word = [0. for _ in range(mem.nwords)]
				for clow in range(mem.nwords):
					if chigh * mem.nwords + clow < channels:
						srcadr = x + y * width + (chigh * mem.nwords + clow) * width * height
						word[clow] = data[srcadr]
					else:
						word[clow] = 0.0
				mem.write(dstadr, word)
	# check
	for y in range(height):
		for x in range(width):
			for c in range(channels):
				goladr = x + y * width + c * width * height
				futadr = x + y * width + (c // mem.nwords) * width * height
				golden = data[goladr]
				fut = mem.read(futadr)[c % mem.nwords]
				assert fut == golden


def mem2vec(mem, width, height, channels):
	out = [0. for _ in range(width * height * channels)]
	for x in range(width):
		for y in range(height):
			for c in range(channels):
				data = mem.read(x + y * width + (c // mem.nwords) * width * height)
				out[x + y * width + c * width * height] = data[c % mem.nwords]
	return out


def test_vec2mem():
	# import importlib
	# importlib.reload(memory)
	# memory.test_vec2mem()
	mem = Memory(2*3*5, 2, verbose=True)
	data = list(range(2*3*5))
	vec2mem(mem, data, 2, 3, 5)
	for a in range(mem.naddr):
		print(a, mem.m[a])


def create_weight_mem_1x1(weights, nwords, channels_in, channels_out):
	WL = nwords
	CI = channels_in
	CU = channels_out
	# create and fill weigth memory
	inblocks = ceil(CI/WL)
	mem = Memory(inblocks * CU, WL)
	for cu in range(CU):
		for ci in range(inblocks):
			data = []
			for c in range(WL):
				srcadr = ci * WL + cu * CI + c
				if srcadr < channels_in * channels_out:
					data.append(weights[srcadr])
				else:
					data.append(0.)
			mem.write(ci + cu*inblocks, data)
	# check
	for ci in range(CI):
		for cu in range(CU):
			golden = weights[ci + cu*CI]
			wut = mem.read(ci // WL + cu * inblocks)[ci % WL]
			assert golden == wut
	return mem


def create_weight_mem_3x3dw(weights, nwords, channels):
	K = 3
	chanblocks = ceil(channels / nwords)
	mem = Memory(K * K * chanblocks, nwords)
	mem.importvec(weights, K, K, channels)
	return mem


	# K = 3
	# WL = nwords
	# C = channels
	# # create and fill weigth memory
	# inblocks = ceil(C/WL)
	# mem = Memory(K * K * inblocks, WL)
	# for h in range(K):
	# 	for w in range(K):
	# 		for chigh in range(inblocks):
	# 			data = []
	# 			for clow in range(WL):
	# 				srcaddr = w + h * K + (clow + chigh * WL) * K * K
	# 				data.append(weights[srcaddr])
	# 			mem.write(w + h * K + chigh * K * K, data)
	# # check
	# for c in channels:
	# 	for h in range(K):
	# 		for w in range(K):
	# 			wut = mem.read(w + h * K + (c // WL))
	# 			golden = weights[w + h * K + c * K * K]
	# 			assert golden == wut[c % WL]
	# return mem


#if  __name__ == 'main':

