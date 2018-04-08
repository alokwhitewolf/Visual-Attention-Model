import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import chainer
from chainer import  serializers
import PIL
from PIL import ImageDraw
import numpy as np
import argparse
import chainer.functions as F
from network import RAM


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAM in Chainer:MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=128,
                        help='Dimension of locator, glimpse hidden state')
    parser.add_argument('--hidden','-hi', type=int, default=256,
                        help='Dimension of lstm hidden state')
    parser.add_argument('--g_size', '-g_size', type=int, default=8,
                        help='Dimension of output')
    parser.add_argument('--len_seq', '-l', type=int, default=6,
                        help='Length of action sequence')
    parser.add_argument('--depth', '-d', type=int, default=1,
                        help='no of depths/glimpses to be taken at once')
    parser.add_argument('--scale', '-s', type=float, default=2,
                        help='subsequent scales of cropped image for sequential depths (int>1)')
    parser.add_argument('--sigma', '-si',type=float, default=0.03,
                        help='sigma of location sampling model')
    parser.add_argument('--eval', '-eval', type=str, default=None,
                        help='Evaluation mode: path to saved model file relative to current working dir')
    args = parser.parse_args()

    model = RAM(args.hidden, args.unit, args.sigma, args.g_size, args.len_seq, args.depth, args.scale)
    serializers.load_npz(os.getcwd() + args.eval, model)
    train, test = chainer.datasets.get_mnist()
    train_data, train_targets = np.array(train).transpose()
    test_data, test_targets = np.array(test).transpose()
    train_data = np.array(list(train_data)).reshape(train_data.shape[0], 1, 28, 28)
    test_data = np.array(list(test_data)).reshape(test_data.shape[0], 1, 28, 28)
    train_targets = np.array(train_targets).astype(np.int32)
    test_targets = np.array(test_targets).astype(np.int32)
    g_size = args.g_size


    def visualize(model):
        chainer.global_config.train = False
        index = np.random.randint(0, 9999)
        x_raw = train_data[index:index + 1]
        t_raw = train_targets[index]
        x = chainer.Variable(np.asarray(x_raw))
        t = chainer.Variable(np.asarray(t_raw))
        batchsize = x.data.shape[0]
        model.reset_state()
        ls = []
        probs = []

        l = np.random.uniform(-1, 1, size=(batchsize, 2)).astype(np.float32)
        l = chainer.Variable(np.asarray(l))
        ls.append(l.data)
        for i in range(6):
            l, ln_pi, y, b = model.forward(x, l, first=True)
            y = F.softmax(y)
            probs.append(y.data)
            ls.append(l.data)
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            ax0 = plt.subplot(gs[0])
            image = PIL.Image.fromarray(train_data[index][0] * 255).convert('RGB')
            canvas = image.copy()
            draw = ImageDraw.Draw(canvas)

            locs = (((ls[i] + 1) / 2) * ((np.array([28, 28])) - 1))

            color = (0, 255, 0)
            xy = np.array([locs[0][0], locs[0][1], locs[0][0], locs[0][1]])
            wh = np.array([-g_size // 2, -g_size // 2, g_size // 2, g_size // 2])
            xys = [xy + np.power(2, s) * wh for s in range(args.depth)]

            for xy in xys:
                draw.rectangle(xy=list(xy), outline=color)
            del draw

            plt.imshow(canvas)
            plt.axis('off')

            y_ticks = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

            bar_lengths = probs[i][0]

            ax1 = plt.subplot(gs[1])
            ax1.barh(y_ticks, bar_lengths, color='#006080')
            ax1.get_xaxis().set_ticks([])
            plt.tight_layout()
            plt.savefig(args.result+str(i) + '.png')


    visualize(model)
