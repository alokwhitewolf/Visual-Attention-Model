import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
from glimpseSensor import getGlimpses


class RAM(chainer.Chain):
    def __init__(self, n_hidden = 256, n_units = 128, n_class = 10, sigma = 0.03, g_size=8, n_steps=6, n_depth=1):
        n_in = g_size * g_size * n_depth
        super(RAM, self).__init__(
            ll1=L.Linear(2, n_units),           # 2 refers to x,y coordinate
            lrho1=L.Linear(n_in, n_units),
            lh1=L.Linear(n_units * 2, n_hidden),
            lh2=L.Linear(n_hidden, n_hidden),
            lstm=L.LSTM(n_hidden, n_hidden),
            la=L.Linear(n_hidden, n_class),     # class/action output
            ll=L.Linear(n_hidden, 2),           # location output
            lb=L.Linear(n_hidden, 1),           # baseline output
        )
        self.sigma = sigma

    def reset_state(self):
        self.lstm.reset_state()





