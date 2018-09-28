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
from dataset import COILDataset


#パス関連
# このファイルの絶対パス
FILE_PATH = Path(__file__).resolve().parent
ROOT_PATH = FILE_PATH.parent.parent
RESULT_PATH = ROOT_PATH.joinpath('results/SeqVAE')
MODEL_PATH = ROOT_PATH.joinpath('models/SeqVAE')

def save_reconstructed_images(x, x1, filename):
    fig, ax = plt.subplots(1, 2, figsize=(18, 9), dpi=100)
    for ai, x in zip(ax.flatten(), (x, x1)):
        x = x.reshape(4, 4, 3, 128, 128).transpose(0, 3, 1, 4, 2).reshape(4 * 128, 4 * 128, 3)
        x = np.clip(x * 255, 0, 255).astype(np.uint8)
        ai.imshow(x)
    fig.savefig(str(filename))
    plt.close()

def save_sampled_images(x, filename):
    x = x.reshape(4, 4,3, 128, 128).transpose(0, 3, 1, 4, 2).reshape(4 * 128, 4 * 128, 3)
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    fig = plt.figure(figsize=(9, 9), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(x)
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
    # Hyper Parameter
    parser.add_argument('--dimz', '-z', default=200, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--coef', '-c', type=float, default=1.0,
                        help='')
    parser.add_argument('--ch', type=int, default=4,
                        help='')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# coef c: {}'.format(args.coef))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    out_path = RESULT_PATH.joinpath('preSeqVAE_latent{}_coef{}_ch{}'.format(args.dimz, args.coef, args.ch))
    print("# result dir : {}".format(out_path))
    out_path.mkdir(exist_ok=True)

    model = SeqVAE(128, args.dimz, args.ch)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(10.0))
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # Load the Idol dataset
    dataset = COILDataset()
    test, train = chainer.datasets.split_dataset(dataset, 200)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,repeat=False, shuffle=False)

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
        save_reconstructed_images(chainer.cuda.to_cpu(x), chainer.cuda.to_cpu(x1),
            out_path.joinpath('train_reconstructed_epoch_{}'.format(trainer.updater.epoch)))

        x = model.xp.array(test[:16])
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1 = model(x).data
        save_reconstructed_images(chainer.cuda.to_cpu(x), chainer.cuda.to_cpu(x1),
            out_path.joinpath('test_reconstructed_epoch_{}'.format(trainer.updater.epoch)))

        # draw images from randomly sampled z
        z1, z2 = np.random.normal(0, 1, (2, args.dimz)).astype(np.float32)
        z = z1 + np.kron(np.linspace(0, 1, 16).astype(np.float32).reshape(16, 1), (z2 - z1))
        x = model.decode(model.xp.asarray(z)).data
        save_sampled_images(chainer.cuda.to_cpu(x),
            out_path.joinpath('sampled_epoch_{}'.format(trainer.updater.epoch)))

    trainer.extend(reconstruct_and_sample)

    # Run the training
    trainer.run()

    chainer.serializers.save_npz(
        str(MODEL_PATH.joinpath('preSeqVAE_latent{}_coef{}_ch{}.npz'.format(args.dimz, args.coef, args.ch))), model)


if __name__ == '__main__':
    main()
