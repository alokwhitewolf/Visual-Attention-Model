import argparse
import numpy as np
import chainer
from chainer import training, datasets, iterators, optimizers, serializers
from chainer import reporter
from network import RAM
from chainer.training import extensions
from weightdecay import lr_drop


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAM in Chainer:MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
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
    parser.add_argument('--evalm', '-evalm', type=str, default=None,
                        help='Evaluation mode: path to saved model file')
    parser.add_argument('--evalo', '-eval0', type=str, default=None,
                        help='Evaluation mode: path to saved optimizer file')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# n_units: {}'.format(args.unit))
    print('# n_hidden: {}'.format(args.hidden))
    print('# Length of action sequence: {}'.format(args.len_seq))
    print('# sigma: {}'.format(args.sigma))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    train, test = chainer.datasets.get_mnist()
    train_data, train_targets = np.array(train).transpose()
    test_data, test_targets = np.array(test).transpose()
    train_data = np.array(list(train_data)).reshape(train_data.shape[0], 1, 28, 28)
    test_data = np.array(list(test_data)).reshape(test_data.shape[0], 1, 28, 28)
    train_targets = np.array(train_targets).astype(np.int32)
    test_targets = np.array(test_targets).astype(np.int32)
    if args.evalm is not None:
        chainer.global_config.train = False

    model = RAM(args.hidden, args.unit, args.sigma,
                 args.g_size, args.len_seq, args.depth, args.scale, using_conv = False)
    #model.to_gpu()
    optimizer = optimizers.NesterovAG()
    if args.evalm is not None:
        serializers.load_npz(args.evalm, model)
        print('model loaded')
    if args.evalo is not None:
        serializers.load_npz(args.evalo, optimizer)
        print('optimizer loaded')

    if args.gpu>=0:
        model.to_gpu()

    optimizer.setup(model)

    train_dataset = datasets.TupleDataset(train_data, train_targets)
    train_iter = iterators.SerialIterator(train_dataset, args.batchsize)
    test_dataset = datasets.TupleDataset(test_data, test_targets)
    train_iter = iterators.SerialIterator(test_dataset, 128)
    stop_trigger = (args.epoch, 'epoch')
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)
    trainer.extend(lr_drop)
    trainer.extend(extensions.snapshot_object(model, '2model{.updater.epoch}.npz'), trigger=(50,'epoch'))
    trainer.extend(extensions.snapshot_object(optimizer, '2opt{.updater.epoch}.npz'), trigger=(50, 'epoch'))
    trainer.extend(extensions.PlotReport(['main/training_accuracy'], 'epoch', trigger=(1, 'epoch'), file_name='2train_accuracy.png',
                          marker="."))
    trainer.extend(extensions.PlotReport(['main/cross_entropy_loss'], 'epoch', trigger=(1, 'epoch'), file_name='2cross_entropy.png',
                          marker="."))
    trainer.extend(extensions.ProgressBar((args.epoch,'epoch'),update_interval=50))
    trainer.run()
