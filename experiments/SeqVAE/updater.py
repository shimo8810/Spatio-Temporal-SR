import chainer
import chainer.functions as F

class SeqVAEUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.enc, self.dec, self.dis = kwargs.pop('models')
        super(SeqVAEUpdater, self).__init__(*args, **kwargs)

    def loss_kld(self):
        pass

    def loss_rec(self):
        pass

    def loss_seq(self):
        pass

    def update_core(self):
        pass


