from math import ceil

def conv1x1_block(xmem, ymem, wmem, width, height, channels_in, channels_out):
	assert xmem.nwords == ymem.nwords == wmem.nwords
	assert width * height * channels_in  // xmem.nwords < xmem.naddr
	assert width * height * channels_out // ymem.nwords < ymem.naddr

	print('conv1x1_block')

	WL = xmem.nwords

	inblocks = ceil(channels_in/WL)
	utblocks = ceil(channels_out/WL)

	print('inblocks', inblocks)
	print('utblocks', utblocks)

	def blockdotprod(xv, fv):
		s = 0
		for x, f in zip(xv, fv):
			s += sum(xi * fi for xi, fi in zip(x, f))
		return s

	for h in range(height):
		for w in range(width):
			x = tuple(xmem.read(w + h*width + c*width*height) for c in range(inblocks))  # fetch all input channels
			for chigh in range(utblocks):
				t = []
				for clow in range(WL):
					cu = chigh*WL + clow # c is output channel
					if cu < channels_out:
						f = tuple(wmem.read(cu*inblocks + c) for c in range(inblocks))
						t.append(blockdotprod(x, f))
					else:
						t.append(0)
				ymem.write(w + h*width + chigh*width*height, t)
