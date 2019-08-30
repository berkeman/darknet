from math import ceil

class nndata():
	def __init__(self, filename):
		self.filename = filename
		self.fh = open(filename, 'rt')

	def nextlayer(self):
		line = self.fh.readline().rstrip('\n').split(' ')
		self.h_in, self.w_in, self.c_in, self.groups, self.k, self.c_out, self.stride = map(int, line[1:])
		line = self.fh.readline().rstrip('\n').split(' ')
		self.h_ut, self.w_ut, self.c_ut, self.pad = map(int, line[1:])
		line = self.fh.readline().rstrip('\n').split(' ')

		def getdata():
			return tuple(map(float, self.fh.readline().rstrip().split(' ')))

		self.weights = getdata()
		self.biases  = getdata()
		self.inputs  = getdata()
		self.outputs = getdata()



def checkconvlayer(N):
	print(N.w_in, N.h_in, N.c_in, '->', N.w_ut, N.h_ut, N.c_ut)
	print(N.k, N.groups, N.stride, N.pad)

	W = N.w_in
	H = N.h_in
	CI = N.c_in
	CU = N.c_ut
	G = N.groups
	P = N.pad
	K = N.k
	S = N.stride

	# @@@ Does not handle groups

	errs = 0
	maxerr = 0

	if G == 1:
		for h in range(0, N.h_in, S):
			for w in range(0, N.w_in, S):
				for cu in range(0, CU):
					t = 0.0
					for ci in range(0, CI//G):
						for y in range(-(K//2), K//2 + 1):
							for x in range(-(K//2), K//2 + 1):
								ww = w + x
								hh = h + y
								if ww < 0 or ww >= W: continue
								if hh < 0 or hh >= H: continue

								six =  ww     + W*hh      + W*H*ci
								wix = (x + P) + K*(y + P) + K*K*ci + K*K*CI*cu
								tix = (w//S) + (h//S)*N.w_ut + (H*W//S//S)*cu
								t += N.inputs[six] * N.weights[wix]
#								print('b %6d %6d %6d     %2d %2d %2d' % (six, wix, tix, g, ci, cu))
					yy = N.outputs[tix]
					if abs(t - yy) >= 1e-5:
						print(w, h, '  ', x, y, '  ', ci, cu, '     ', t, yy)
						errs += 1
						maxerr = max(maxerr, abs(t-yy))
	elif G == CI == CU:
		pass
	else:
		assert False


class Memory():
	def __init__(self, naddr, nwords):
		print('[Memory create: %d x %d]' % (naddr, nwords,))
		self.m = [[0 for x in range(nwords)] for y in range(naddr)]
		self.naddr = naddr
		self.nwords = nwords

	def write(self, a, d):
		assert len(d) == self.nwords
		assert 0 <= a <= self.naddr
		self.m[a] = d
#		print('write %5d' % (a,), ''.join("% 5.2f" % x for x in d))

	def read(self, a):
		assert 0 <= a <= self.naddr
		return self.m[a]



def convlayer(N):
	print(N.w_in, N.h_in, N.c_in, '->', N.w_ut, N.h_ut, N.c_ut)
	print(N.k, N.groups, N.stride, N.pad)

	W = N.w_in
	H = N.h_in
	CI = N.c_in
	CU = N.c_ut
	G = N.groups
	P = N.pad
	K = N.k
	S = N.stride

	out = [0. for _ in range(H*W*CU//S//S)]

	if G == 1:
		for h in range(0, H, S):
			for w in range(0, W, S):
				for cu in range(0, CU):
					t = 0.0
					for ci in range(0, CI):
						for y in range(-(K//2), K//2 + 1):
							for x in range(-(K//2), K//2 + 1):
								ww = w + x
								hh = h + y
								if ww < 0 or ww >= W: continue
								if hh < 0 or hh >= H: continue

								six =  ww     + W*hh      + W*H*ci
								wix = (x + P) + K*(y + P) + K*K*ci + K*K*CI*cu
								tix = (w//S) + (h//S)*N.w_ut + (H*W//S//S)*cu
								t += N.inputs[six] * N.weights[wix]
					out[tix] = t
	elif G == CI == CU:
		for h in range(0, H, S):
			for w in range(0, W, S):
				ci = 0
				for cu in range(0, CU):
					t = 0.0
					for y in range(-(K//2), K//2 + 1):
						for x in range(-(K//2), K//2 + 1):
							ww = w + x
							hh = h + y
							if ww < 0 or ww >= W: continue
							if hh < 0 or hh >= H: continue

							six =  ww     + W*hh      + W*H*cu
							wix = (x + P) + K*(y + P) + K*K*ci + K*K*cu
							tix = (w//S) + (h//S)*N.w_ut + (H*W//S//S)*cu
#							print('dw %6d %6d %3d %3d   %3d %3d   %3d %3d   %f %f' % (six, wix, w, h, ww, hh, ci, cu, N.inputs[six], N.weights[wix]))
							t += N.inputs[six] * N.weights[wix]
#					print(t, N.outputs[tix])
					out[tix] = t
		pass
	else:
		assert False
	return out



def check(out, N):
	errs = 0
	maxerr = 0
	cnt = 0
	for ix in range(N.h_ut*N.w_ut*N.c_ut):
		golden = N.outputs[ix]
		pred = out[ix]
		diff = abs(golden - pred)
		maxerr = max(maxerr, diff)
		if diff >= 1e-5:
#			print(ix, golden, pred, diff)
			errs += 1
		cnt += 1

	print('checked', cnt)
	print('errs   ', errs)
	print('maxerr ', maxerr)


def create_weight_mem(N):
	WL = xmem.nwords

	CI = N.c_in
	CU = N.c_ut

	# create and fill weigth memory
	inblocks = ceil(CI/WL)
	wmem = Memory(inblocks*CU, WL)
	for cu in range(CU):
		for ci in range(inblocks):
			data = []
			for c in range(WL):
				srcadr = ci*WL + cu*CI + c
				if srcadr < N.c_in * N.c_ut:
					data.append(N.weights[srcadr])
				else:
					data.append(0.)
			wmem.write(ci + cu*inblocks, data)
	# test it
	for ci in range(CI):
		for cu in range(CU):
			golden = N.weights[ci + cu*CI]
			wut = wmem.read(ci//WL + cu*inblocks)[ci%WL]
			assert golden == wut
	return wmem



def conv1x1(xmem, ymem, N):
	assert N.k == 1
	assert N.groups == 1
	assert N.h_in * N.w_in * N.c_in // xmem.nwords <= xmem.naddr
	assert xmem.nwords == ymem.nwords

	print('conv1x1')
	print('   ', N.w_in, N.h_in, N.c_in, '->', N.w_ut, N.h_ut, N.c_ut)
	print('   ', N.k, N.groups, N.stride, N.pad)

	WL = xmem.nwords

	W = N.w_in
	H = N.h_in
	CI = N.c_in
	CU = N.c_ut

	wmem = create_weight_mem(N)

	inblocks = ceil(CI/WL)
	utblocks = ceil(CU/WL)

	print('inblocks', inblocks)
	print('utblocks', utblocks)


	for h in range(H):
		for w in range(W):
			for cu in range(utblocks):
				temp = [0 for _ in range(WL)]
				for c in range(WL):
					if cu*WL + c >= CU:
						# num output channels not divisible by WL
						continue
					d = [0. for _ in range(WL)]
					for ci in range(inblocks):
						data = xmem.read(w + h*W + ci*W*H)
						weight = wmem.read(ci + (c + cu*WL)*inblocks)
						d = [x * y + z for x, y, z in zip(data, weight, d)]
					temp[c] = sum(d)
				ymem.write(w + h*W + cu*W*H, temp)






def printv(x):
	print(' '.join("%5.2f" % (c,) for c in x))





def conv1x1_block(xmem, ymem, N):
	assert N.k == 1
	assert N.groups == 1
	assert N.h_in * N.w_in * N.c_in // xmem.nwords <= xmem.naddr
	assert xmem.nwords == ymem.nwords

	print('conv1x1_block')
	print('   ', N.w_in, N.h_in, N.c_in, '->', N.w_ut, N.h_ut, N.c_ut)
	print('   ', N.k, N.groups, N.stride, N.pad)

	WL = xmem.nwords

	W = N.w_in
	H = N.h_in
	CI = N.c_in
	CU = N.c_ut

	wmem = create_weight_mem(N)

	inblocks = ceil(CI/WL)
	utblocks = ceil(CU/WL)

	print('inblocks', inblocks)
	print('utblocks', utblocks)

	def blockdotprod(xv, fv):
		s = 0
		for x, f in zip(xv, fv):
			s += sum(xi * fi for xi, fi in zip(x, f))
		return s

	for h in range(H):
		for w in range(W):
			x = tuple(xmem.read(w + h*W + c*H*W) for c in range(inblocks))  # fetch all input channels
			for chigh in range(utblocks):
				t = []
				for clow in range(WL):
					cu = chigh*WL + clow # c is output channel
					if cu < CU:
						f = tuple(wmem.read(cu*inblocks + c) for c in range(inblocks))
						t.append(blockdotprod(x, f))
					else:
						t.append(0)
				ymem.write(w + h*W + chigh*W*H, t)





















N = nndata('darknet_run.txt')

def feat2mem(mem, N):
	assert N.w_in * N.h_in * N.c_in // mem.nwords <= mem.naddr
	depth = ceil(N.c_in / mem.nwords)
	for y in range(N.w_in):
		for x in range(N.h_in):
			for chigh in range(depth):
				dstadr = x + y*N.w_in + chigh*N.w_in*N.h_in
				word = [0. for _ in range(mem.nwords)]
				for clow in range(mem.nwords):
					if chigh*mem.nwords+clow < N.c_in:
						srcadr = x + y*N.w_in + (chigh*mem.nwords+clow)*N.w_in*N.h_in
						word[clow] = N.inputs[srcadr]
					else:
						word[clow] = 0.0
				mem.write(dstadr, word)

	for y in range(N.w_in):
		for x in range(N.h_in):
			for c in range(N.c_in):
				goladr = x + y*N.w_in + c*N.w_in*N.h_in
				futadr = x + y*N.w_in + (c//mem.nwords)*N.w_in*N.h_in
				golden = N.inputs[goladr]
				fut = mem.read(futadr)[c%mem.nwords]
				assert fut == golden



def store(N, nwords):
	m = Memory(N.w_in * N.w_ut, nwords)
	assert N.c_in <= nwords
	for y in range(N.h_in):
		for x in range(N.w_in):
			t = []
			for c in range(nwords):
				t.append(N.inputs[x + y*N.w_in + c*N.w_in*N.h_in])
			m.write(x + y*N.w_in, t)
	return m

def unstore(N, mem):
#	assert N.c_ut <= mem.nwords
	out = [0. for _ in range(N.w_ut*N.h_ut*N.c_ut)]
	for x in range(N.w_ut):
		for y in range(N.h_ut):
			for c in range(N.c_ut):
				data = mem.read(x + y*N.w_ut + c//mem.nwords*N.w_ut*N.h_ut)
				out[x + y*N.w_ut + c*N.w_ut*N.h_ut] = data[c%mem.nwords]
	return out


WL = 32

xmem = Memory(112*112*2, WL)
ymem = Memory(112*112*5, WL)

for x in range(50):
	print()
	print(x)
	N.nextlayer()
#	if N.c_in <= WL and N.c_ut <= WL and N.k == 1:
	if N.k == 1:
		print(N.w_in, N.h_in, N.c_in, '->', N.w_ut, N.h_ut, N.c_ut)
		print(N.k, N.groups, N.stride, N.pad)

		feat2mem(xmem, N)
		conv1x1_block(xmem, ymem, N)
		out = unstore(N, ymem)
		check(out, N)



	else:
		continue
		print('conv')
		out = convlayer(N)
		print('check')
		check(out, N)
		print('')
