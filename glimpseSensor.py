import numpy as np
from chainer import cuda
from chainer import function
from chainercv.transforms import resize



class GlimpseSensor(function.Function):
	def __init__(self, center, output_size, scale=1):
		if type(output_size) is not tuple:
			self.output_size = output_size
		else:
			assert output_size[0] == output_size[1],"Output dims must be same"
			self.output_size = output_size[0]
		self.center = center
		self.scale = scale

	def forward(self, images):
		xp = cuda.get_array_module(*images)
		n, c, h_i, w_i = images.shape
		assert h_i == w_i, "Image should be square"
		size_i = h_i
		size_o = self.output_size

		# [-1, 1]^2 -> [0, size_i - 1]x[0, size_i - 1]

		center = 0.5 * (self.center + 1) * (size_i - 1)  # center -> [n X 2]

		y = xp.zeros(shape=(n, c*self.scale, size_o, size_o), dtype=np.float32)

		xmin = xp.zeros(shape=(self.scale, n), dtype=np.int32)
		ymin = xp.zeros(shape=(self.scale, n), dtype=np.int32)
		xmax = xp.zeros(shape=(self.scale, n), dtype=np.int32)
		ymax = xp.zeros(shape=(self.scale, n), dtype=np.int32)

		for scale in range(self.scale):
			xmin[scale] = xp.round(xp.clip(center[:, 0] - (0.5 * size_o * (scale+1)), 0, size_i - 1))
			ymin[scale] = xp.round(xp.clip(center[:, 1] - (0.5 * size_o * (scale+1)), 0, size_i - 1))
			xmax[scale] = xp.round(xp.clip(center[:, 0] + (0.5 * size_o * (scale+1)), 0, size_i - 1))
			ymax[scale] = xp.round(xp.clip(center[:, 1] + (0.5 * size_o * (scale+1)), 0, size_i - 1))

		for i in range(n):
			for j in range(self.scale):
				cropped = images[i][:,xmin[j][i]:xmax[j][i]+1, ymin[j][i]:ymax[j][i]+1]
				resized = resize(cropped, (self.output_size, self.output_size))
				y[i][c*j: (c*j)+c] = resized

		return y,

	def backward(self, images, gy):
		#return zero grad
		xp = cuda.get_array_module(*images)
		n, c_out = gy[0].shape[:2]
		c_in ,h_i, w_i = images.shape[1:4]
		gx = xp.zeros(shape=(n, c_in, h_i, w_i), dtype=np.float32)
		return gx,

def getGlimpses(x, center, size, scale=1):
	return GlimpseSensor(center, size, scale)(x)