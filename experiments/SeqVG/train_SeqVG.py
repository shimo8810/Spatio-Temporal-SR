"""
train fi with SeqVAE-GAN
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

from net import Encoder, Decoder, Discriminator
from dataset import SeqCOILDataset, SeqMovingMNISTDataset
from updater import SeqVGUpdater

#パス関連
# このファイルの絶対パス
FILE_PATH = Path(__file__).resolve().parent
ROOT_PATH = FILE_PATH.parent.parent
RESULT_PATH = ROOT_PATH.joinpath('results/SeqVG')
MODEL_PATH = ROOT_PATH.joinpath('models/SeqVG')

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
    parser.add_argument('--epoch', '-e', default=100, type=int,
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
    parser.add_argument('--coef3', type=float, default=1.0,
                        help='')
    parser.add_argument('--coef4', type=float, default=0.01,
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
    print('# reconstruction Loss coef: {}'.format(args.coef1))
    print('# Sequence Loss coef: {}'.format(args.coef2))
    print('# KL-divergence Loss coef: {}'.format(args.coef3))
    print('# adversal Loss coef: {}'.format(args.coef4))
    print('')

    out_path = RESULT_PATH.joinpath('{}/SeqVG_latent{}_ch{}_coef1{}_coef2{}_coef3{}_coef4{}'.format(
        args.dataset, args.latent, args.ch, args.coef1, args.coef2, args.coef3, args.coef4))
    print("# result dir : {}".format(out_path))
    out_path.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'coil':
        data_ch = 3
        data_size = 128
    elif args.dataset == 'mmnist':
        data_ch = 1
        data_size = 64

    enc = Encoder(data_size, data_ch, args.latent, args.ch)
    dec = Decoder(data_size, data_ch, args.latent, args.ch)
    dis = Discriminator(args.ch)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        enc.to_gpu()
        dec.to_gpu()
        dis.to_gpu()

    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha, beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(10e-4))
        return optimizer

    opt_enc = make_optimizer(enc)
    opt_dec = make_optimizer(dec)
    opt_dis = make_optimizer(dis)

    # Load the Idol dataset
    if args.dataset == 'coil':
        dataset = SeqCOILDataset()
        test, train = chainer.datasets.split_dataset(dataset, 200)
    elif args.dataset == 'mmnist':
        test = SeqMovingMNISTDataset(dataset='test')
        train = SeqMovingMNISTDataset(dataset='train')

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = SeqVGUpdater(
        models=(enc, dec, dis),
        iterator={
            'main': train_iter,
            'test': test_iter
        },
        optimizer={
            'enc': opt_enc,
            'dec': opt_dec,
            'dis': opt_dis
        },
        K=1,
        coefs=(args.coef1, args.coef2, args.coef3, args.coef4),
        device=args.gpu
    )

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=str(out_path))
    # trainer.extend(extensions.Evaluator(
    #     test_iter, model, device=args.gpu, eval_func=model.get_seq_loss_func(C1=args.coef1, C2=args.coef2, k=10)))
    # trainer.extend(extensions.ExponentialShift("alpha", 0.1), trigger=(50, 'epoch'))
    trainer.extend(extensions.dump_graph('dec/loss'))
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'dec/loss', 'enc/loss', 'dis/loss', 'elapsed_time']))

    trainer.extend(extensions.PlotReport(['dec/loss'],
                              'epoch', file_name='dec_loss.png'))
    trainer.extend(extensions.PlotReport(['enc/loss'],
                              'epoch', file_name='enc_loss.png'))
    trainer.extend(extensions.PlotReport(['dis/loss'],
                              'epoch', file_name='dis_loss.png'))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training, and I will get a cup of tea.
    trainer.run()

    model_save_path = MODEL_PATH.joinpath(args.dataset, 'Decoder_SeqVG_latent{}_ch{}_coef1{}_coef2{}_coef3{}_coef4{}.npz'.format(
            args.latent, args.ch, args.coef1, args.coef2, args.coef3, args.coef4))
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    chainer.serializers.save_npz(str(model_save_path), dec)

    model_save_path = MODEL_PATH.joinpath(args.dataset, 'Encoder_SeqVG_latent{}_ch{}_coef1{}_coef2{}_coef3{}_coef4{}.npz'.format(
            args.latent, args.ch, args.coef1, args.coef2, args.coef3, args.coef4))
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    chainer.serializers.save_npz(str(model_save_path), enc)

    model_save_path = MODEL_PATH.joinpath(args.dataset, 'Discriminater_SeqVG_latent{}_ch{}_coef1{}_coef2{}_coef3{}_coef4{}.npz'.format(
            args.latent, args.ch, args.coef1, args.coef2, args.coef3, args.coef4))
    print(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    chainer.serializers.save_npz(str(model_save_path), dis)

if __name__ == '__main__':
    main()
