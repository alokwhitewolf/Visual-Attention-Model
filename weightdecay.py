from chainer import training

@training.make_extension(trigger=(400, 'epoch'))
def lr_drop(trainer):
    trainer.updater.get_optimizer('main').lr *= 0.1