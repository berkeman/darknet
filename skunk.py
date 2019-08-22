
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



N = nndata('darknet_run.txt')

N.nextlayer()
N.nextlayer()

print(N.w_in, N.h_in, N.c_in, '->', N.w_ut, N.h_ut, N.c_ut)
print(N.k, N.stride, N.pad)
print()

# @@@ Still only one output channel
# @@@ Does not handle groups

for h in range(0, N.h_in, N.stride):
	for w in range(0, N.w_in, N.stride):
		t = 0.0
		for cc in range(0, N.c_in): # channel
			for x in range(-(N.k//2), N.k//2 + 1):
				for y in range(-(N.k//2), N.k//2 + 1):
					ww = w + x
					hh = h + y
					if ww < 0 or ww > N.w_in: continue
					if hh < 0 or hh > N.h_in: continue
					t += N.inputs[ww + hh*N.w_in + cc*N.w_in*N.h_in] * N.weights[x + N.pad + N.k*(y + N.pad) +cc*N.k*N.k]
					print(w, h, x, y, cc, ww + hh*N.w_in + cc*N.w_in*N.h_in, x + N.pad + N.k*(y + N.pad) +cc*N.k*N.k)

		y = N.outputs[w//N.stride + (h//N.stride)*N.w_ut]
		print(w, h, t, y, t-y, abs(t-y) < 1e-5)
