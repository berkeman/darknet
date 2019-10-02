from math import ceil, sqrt, nan

class Blockdotprod():
	def __init__(self):
		self.mulcnt = 0
		self.cccnt = 0
		pass
	def mac(self, xvec, kvec):
		s = 0
		for x, k in zip(xvec, kvec):
			s += sum(xi * ki for xi, ki in zip(x, k))
		self.mulcnt += sum(len(x) for x in xvec)
		self.cccnt += len(xvec)
		return s
	def status(self):
		return(dict(mulcnt=self.mulcnt, cccnt=self.cccnt))

class BiasNorm():
	def __init__(self, layer):
		self.mean = layer.rolling_mean
		self.var  = layer.rolling_var # var may be negative from Darknet!
		self.bias = layer.biases
		self.scales = layer.scales
		self.activation = layer.activation
		self.batchnorm = layer.bn
	def bn(self, x, channel):
		# Bias and (batch) normalisation
		# x = input value
		if channel >= len(self.bias):
			# @@@ NB: Emulating word alignment, pad with zeros
			return 0
		if self.batchnorm:
			if self.var[channel] < 0:
				# This is what happens inside Darknet,
				# we do the same thing to keep our results
				# as close as possible to Darknet.
				x = nan
			else:
				x = (x - self.mean[channel]) / (sqrt(self.var[channel]) + .000001) # eps from blas.c
			x = x * self.scales[channel]
			x = x + self.bias[channel]
		else:
			x = x + self.bias[channel]

		if self.activation == 'relu':
			return max(0, x)
		else:
			assert self.activation == 'linear'
			return x



class Conv1x1_block():
	def __init__(self, xreadfun, wmem, width, height, channels_in, channels_out, bias, WL, name=''):
		self.name = name
		self.xreadfun = xreadfun
		self.wmem = wmem
		self.bias = bias
		self.width = width
		self.height = height
		self.channels_in = channels_in
		self.channels_out = channels_out
		self.WL = WL
		self.inblocks = ceil(channels_in/WL)
		self.utblocks = ceil(channels_out/WL)
		self.bdp = Blockdotprod()
	def conv(self, w, h):
		""" get full 1x1 convolution output at spatial coords (w, h) """
		res = []
		# fetch all input channels/blocks at once
		x = tuple(self.xreadfun((w, h, c)) for c in range(self.inblocks))
		for chigh in range(self.utblocks):
			t = []
			for clow in range(self.WL):
				cu = chigh*self.WL + clow # c is output channel
				if cu < self.channels_out:
					# fetch all coeff for one output channel
					f = tuple(self.wmem.read(cu*self.inblocks + c) for c in range(self.inblocks))
					y = self.bdp.mac(x, f)
					y = self.bias.bn(y, cu)
					t.append(y)
				else:
					t.append(0)
			res.append(t)
		return res
	def status(self):
		return dict(
			typ='1x1',
			macs=self.bdp.status()['mulcnt'],
			cc=self.bdp.status()['cccnt'],
			wi=self.width,
			hi=self.height,
			wo=self.width,
			ho=self.height,
			ci=self.channels_in,
			co=self.channels_out,
			stride=1,
			WL=self.WL,
		)

class Conv3x3dw_block():
	def __init__(self, xreadfun, wmem, width, height, channels, stride, bias, WL=32, name=''):
		self.name = name
		self.xreadfun = xreadfun
		self.wmem = wmem
		self.bias = bias
		self.width = width
		self.height = height
		self.channels = channels
		self.WL = WL
		self.chanblocks = ceil(channels/WL)
		self.cc = 0
		self.macs = 0
		self.macscc = 0
		self.stride = stride
		print('3x3 stride', stride)
	def conv(self, w, h):
		K = 3
		K2 = K//2
		res = []
		w *= self.stride
		h *= self.stride
		for chigh in range(self.chanblocks):
			acc = [0 for _ in range(self.WL)]
			for y in range(-K2, K2+1):
				for x in range(-K2, K2+1):
					wadr = (x+K2) + (y+K2) * K + chigh * K * K
					if x + w < 0 or x + w >= self.width or y + h < 0 or y + h >= self.height:
						data = [0 for _ in range(self.WL)]
					else:
						data = self.xreadfun((w+x, h+y, chigh))
					weight = self.wmem.read(wadr)
					acc = [ta + tw * tx for ta, tw, tx in zip(acc, weight, data)]
			self.cc += K * K
			self.macs += K * K * self.WL
			acc = tuple(self.bias.bn(x, chigh * self.WL + ix) for ix, x in enumerate(acc))
			res.append(acc)
		return res
	def status(self):
		return dict(
			typ='3x3dw',
			macs=self.macs,
			cc=self.cc,
			wi=self.width,
			hi=self.height,
			wo=self.width//self.stride,
			ho=self.height//self.stride,
			ci=self.channels,
			co=self.channels,
			stride=self.stride,
			WL=self.WL,
)
