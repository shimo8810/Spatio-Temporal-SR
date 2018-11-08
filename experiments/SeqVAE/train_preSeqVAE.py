"""
train fi with vae
"""
from pathlib import Path
import argparse
import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')

from tqdm import tqdm
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import chainer
from chainer import training
from chainer.training import extensions

from net import SeqVAE
from dataset import COILDataset, MovingMNISTDataset


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
    parser.add_argument('--latent', '-l', default=200, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--coef', '-c', type=float, default=1.0,
                        help='')
    parser.add_argument('--ch', type=int, default=4,
                        help='')
    args = parser.parse_args()

    print('Dataset: {}'.format(args.dataset))
    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.latent))
    print('# coef c: {}'.format(args.coef))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    out_path = RESULT_PATH.joinpath('{}/preSeqVAE_latent{}_coef{}_ch{}'.format(args.dataset, args.latent, args.coef, args.ch))
    print("# result dir : {}".format(out_path))
    out_path.mkdir(parents=True, exist_ok=True)

    # Load the Idol dataset
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
        dataset = COILDataset()
        test, train = chainer.datasets.split_dataset(dataset, 200)
    elif args.dataset == 'mmnist':
        test = MovingMNISTDataset(dataset='test')
        train = MovingMNISTDataset(dataset='train')

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(
        train_iter, optimizer,
        device=args.gpu, loss_func=model.get_loss_func(C=args.coef))

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=str(out_path))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu, eval_func=model.get_loss_func(C=args.coef, k=10)))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/rec_loss', 'validation/main/rec_loss', 'main/kl_loss', 'validation/main/kl_loss', 'elapsed_time']))

    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'))
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
        x = model.xp.array(train[:16])
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1 = model(x).data
        save_reconstructed_images(model.xp.asnumpy(x), model.xp.asnumpy(x1),
            out_path.joinpath('train_reconstructed_epoch_{}'.format(trainer.updater.epoch)), data_ch, data_size)

        x = model.xp.array(test[:16])
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1 = model(x).data
        save_reconstructed_images(model.xp.asnumpy(x), model.xp.asnumpy(x1),
            out_path.joinpath('test_reconstructed_epoch_{}'.format(trainer.updater.epoch)), data_ch, data_size)

    trainer.extend(reconstruct_and_sample)

    # Run the training
    trainer.run()

    model_save_path = MODEL_PATH.joinpath('{}/preSeqVAE_latent{}_coef{}_ch{}.npz'.format(args.dataset, args.latent, args.coef, args.ch))
    model_save_path.mkdir(parents=True, exist_ok=True)
    chainer.serializers.save_npz(str(model_save_path), model)


if __name__ == '__main__':
    main()
