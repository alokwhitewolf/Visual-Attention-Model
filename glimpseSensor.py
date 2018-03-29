import numpy as np
import chainer
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

	def forward(self, image):
		xp = cuda.get_array_module(*image)
		n, c, h_i, w_i = image.shape
		assert h_i == w_i, "Image should be square"
		size_i = h_i
		size_o = self.output_size

		# [-1, 1]^2 -> [0, size_i - 1]x[0, size_i - 1]
		center = 0.5 * (self.center + 1) * (size_i - 1)  # center -> [n X 2]

		y = xp.zeros(shape=(n, c*self.scale, size_o, size_o), dtype=np.float32)

		xmin = xp.zeros(shape=(self.scale, n), dtype=np.float32)
		ymin = xp.zeros(shape=(self.scale, n), dtype=np.float32)
		xmax = xp.zeros(shape=(self.scale, n), dtype=np.float32)
		ymax = xp.zeros(shape=(self.scale, n), dtype=np.float32)

		for scale in range(self.scale):
			xmin[scale] = xp.round(xp.clip(center[:, 0] - (0.5 * size_o * (scale+1)), 0, size_i - 1))
			ymin[scale] = xp.round(xp.clip(center[:, 1] - (0.5 * size_o * (scale+1)), 0, size_i - 1))
			xmax[scale] = xp.round(xp.clip(center[:, 0] + (0.5 * size_o * (scale+1)), 0, size_i - 1))
			ymax[scale] = xp.round(xp.clip(center[:, 1] + (0.5 * size_o * (scale+1)), 0, size_i - 1))

		for i in range(n):
			for j in range(self.scale):
				cropped = image[i][:,xmin[j][i]:xmax[j][i]+1, ymin[j][i]:ymax[j][i]+1]
				resized = resize(cropped, (self.output_size, self.output_size))
				y[i][c*j: (c*j)+c] = resized

		return y,

