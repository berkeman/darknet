
class nndata():
	def __init__(self, filename):
		self.filename = filename
		self.fh = open(filename, 'rt')

	def nextlayer(self):
		line = self.fh.readline().rstrip('\n').split(' ')
		self.h_in, self.w_in, self.c_in, self.groups, self.ksize, self.c_out, self.stride = map(int, line[1:])
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
print(N.biases)
print(N.stride)

x = N.inputs
f = N.weights
L = N.h_in*N.w_in
print(N.h_in, N.w_in, len(x), 4+2*L)

print('x', N.h_in, N.h_ut, N.c_in, N.c_ut)

y = x[4]*f[4] + x[5]*f[5] + x[7]*f[7] + x[8]*f[8] +\
x[4+L]*f[13] + x[5+L]*f[14] + x[7+L]*f[16] + x[8+L]*f[17] +\
x[4+2*L]*f[22] + x[5+2*L]*f[23] + x[7+2*L]*f[25] + x[8+2*L]*f[26]

print(y)
print(N.outputs[0])

# for h in range(0, N.h_in, N.stride):
# 	for w in range(0, N.w_in, N.stride):
# 		for n in range(0, N.c_in):
# 			t = 0.0
# 			for hh in range(N.ksize):
# 				for ww in range(N.ksize):
# 					for cc in range(N.c_in):
# 						pass
