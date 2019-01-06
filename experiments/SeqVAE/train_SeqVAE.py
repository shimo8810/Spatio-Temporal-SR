"""
train fi with vae
"""
from pathlib import Path
import argparse
import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import training
from chainer.training import extensions

from net import SeqVAE
from dataset import SeqCOILDataset, SeqMovingMNISTDataset

#パス関連
# このファイルの絶対パス
FILE_PATH = Path(__file__).resolve().parent
ROOT_PATH = FILE_PATH.parent.parent
RESULT_PATH = ROOT_PATH.joinpath('results/SeqVAE')
MODEL_PATH = ROOT_PATH.joinpath('models/SeqVAE')

def save_reconstructed_images(x, x1, filename, data_ch, data_size):
    fig, ax = plt.subplots(1, 2, figsize=(18, 9), dpi=100)
    for ai, x in zip(ax.flatten(), (x, x1)):
        x = x.reshape(4, 4, data_ch, data_size, data_size).transpose(0, 3, 1, 4, 2).reshape(4 * data_size, 4 * data_size, data_ch)
        if data_ch == 1:
            x = x.reshape(4 * data_size, 4 * data_size)
        x = np.clip(x * 255, 0, 255).astype(np.uint8)
        ai.imshow(x)
    fig.savefig(str(filename))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=200, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='learning minibatch size')
    parser.add_argument('--dataset', '-d', type=str, choices=['coil', 'mmnist'], default='mmnist',
                        help='using dataset')
    # Hyper Parameter
    parser.add_argument('--latent', '-l', default=100, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--coef1', type=float, default=1.0,
                        help='')
    parser.add_argument('--coef2', type=float, default=0.5,
                        help='')
    parser.add_argument('--ch', type=int, default=4,
                        help='')
    args = parser.parse_args()

    print('### Learning Parameter ###')
    print('# Dataset: {}'.format(args.dataset))
    print('# GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('### Model Parameter ###')
    print('# dimension of latent z: {}'.format(args.latent))
    print('# channel scale: {}'.format(args.ch))
    print('# KL Loss coef: {}'.format(args.coef1))
    print('# Seq Loss coef: {}'.format(args.coef2))
    print('')

    out_path = RESULT_PATH.joinpath('{}/SeqVAE_epoch{}_latent{}_ch{}_coef1{}_coef2{}'.format(
        args.dataset, args.epoch, args.latent, args.ch, args.coef1, args.coef2))
    print("# result dir : {}".format(out_path))
    out_path.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'coil':
        data_ch = 3
        data_size = 128
    elif args.dataset == 'mmnist':
        data_ch = 1
        data_size = 64

    model = SeqVAE(data_size, data_ch, args.latent, args.ch)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # Load the Idol dataset
    if args.dataset == 'coil':
        dataset = SeqCOILDataset()
        test, train = chainer.datasets.split_dataset(dataset, 200)
    elif args.dataset == 'mmnist':
        test = SeqMovingMNISTDataset(dataset='test')
        train = SeqMovingMNISTDataset(dataset='train')

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(
        train_iter, optimizer,
        device=args.gpu, loss_func=model.get_seq_loss_func(C1=args.coef1, C2=args.coef2))

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=str(out_path))
    trainer.extend(extensions.Evaluator(
        test_iter, model, device=args.gpu, eval_func=model.get_seq_loss_func(C1=args.coef1, C2=args.coef2, k=10)))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/rec_loss', 'validation/main/rec_loss',
        'main/seq_loss', 'validation/main/seq_loss', 'main/kl_loss', 'validation/main/kl_loss', 'elapsed_time']))

    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/seq_loss', 'validation/main/seq_loss'],
                              'epoch', file_name='seq_loss.png'))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/rec_loss'],
                              'epoch', file_name='rec_loss.png'))
    trainer.extend(extensions.PlotReport(['main/kl_loss', 'validation/main/kl_loss'],
                              'epoch', file_name='kl_loss.png'))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    #draw reconstructred image
    @chainer.training.make_extension(trigger=(10, 'epoch'))
    def reconstruct_and_sample(trainer):
        x = model.xp.array(train[:16])[:,0,:]
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1 = model(x).data
        save_reconstructed_images(model.xp.asnumpy(x), model.xp.asnumpy(x1),
            out_path.joinpath('train_reconstructed_epoch_{}'.format(trainer.updater.epoch)), data_ch, data_size)

        x = model.xp.array(test[:16])[:,0,:]
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1 = model(x).data
        save_reconstructed_images(model.xp.asnumpy(x), model.xp.asnumpy(x1),
            out_path.joinpath('test_reconstructed_epoch_{}'.format(trainer.updater.epoch)), data_ch, data_size)

    trainer.extend(reconstruct_and_sample)

    # Run the training, and I will get a cup of tea.
    trainer.run()

    model_save_path = MODEL_PATH.joinpath(args.dataset, 'SeqVAE_epoch{}_latent{}_ch{}_coef1{}_coef1{}.npz'.format(
            args.epoch, args.latent, args.ch, args.coef1, args.coef2))
    print(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    chainer.serializers.save_npz(str(model_save_path), model)

if __name__ == '__main__':
    main()
