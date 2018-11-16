import chainer
import chainer.functions as F

class SeqVGUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.enc, self.dec, self.dis = kwargs.pop('models')
        self.K = kwargs.pop('K')
        self.c1, self.c2, self.c3, self.c4 = kwargs.pop('coefs')
        super(SeqVGUpdater, self).__init__(*args, **kwargs)

    def enc_loss(self, enc, x1, x2, x3, mu1, mu3, ln_var1, ln_var3, x1_rec, x2_rec, x3_rec, y1_fake, y2_fake, y3_fake):
        batchsize = x1.shape[0]
        rec_loss = 0
        seq_loss = 0
        kl_loss = 0
        adv_loss = 0
        # rec loss
        rec_loss += F.bernoulli_nll(x1, x1_rec) / 2.0 / batchsize
        rec_loss += F.bernoulli_nll(x3, x3_rec) / 2.0 / batchsize
        # kl_loss
        kl_loss += F.gaussian_kl_divergence(mu1, ln_var1) / 2.0 / batchsize
        kl_loss += F.gaussian_kl_divergence(mu3, ln_var3) / 2.0 / batchsize
        # seq loss
        seq_loss += F.bernoulli_nll(x2, x2_rec) / batchsize
        # adv_loss
        adv_loss += F.sum(F.softplus(-y1_fake)) / 3.0 / batchsize
        adv_loss += F.sum(F.softplus(-y2_fake)) / 3.0 / batchsize
        adv_loss += F.sum(F.softplus(-y3_fake)) / 3.0 / batchsize

        loss = self.c1 * rec_loss + self.c2 * seq_loss + self.c3 * kl_loss + self.c4 * adv_loss
        chainer.report({'loss': loss}, enc)
        return loss

    def dec_loss(self, dec, x1, x2, x3, x1_rec, x2_rec, x3_rec, y1_fake, y2_fake, y3_fake):
        batchsize = x1.shape[0]
        rec_loss = 0
        seq_loss = 0
        adv_loss = 0
        # rec loss
        rec_loss += F.bernoulli_nll(x1, x1_rec) / 2.0 / batchsize
        rec_loss += F.bernoulli_nll(x3, x3_rec) / 2.0 / batchsize
        # seq loss
        seq_loss += F.bernoulli_nll(x2, x2_rec) / batchsize
        # adv_loss
        adv_loss += F.sum(F.softplus(-y1_fake)) / 3.0 / batchsize
        adv_loss += F.sum(F.softplus(-y2_fake)) / 3.0 / batchsize
        adv_loss += F.sum(F.softplus(-y3_fake)) / 3.0 / batchsize

        loss = self.c1 * rec_loss + self.c2 * seq_loss + self.c4 * adv_loss
        chainer.report({'loss': loss}, dec)
        return loss

    def dis_loss(self, dis, y1_fake, y2_fake, y3_fake, y1_real, y2_real, y3_real):
        batchsize = y1_fake.shape[0]

        fake_loss = 0
        real_loss = 0

        fake_loss += F.sum(F.softplus(y1_fake)) / 3.0 / batchsize
        fake_loss += F.sum(F.softplus(y2_fake)) / 3.0 / batchsize
        fake_loss += F.sum(F.softplus(y3_fake)) / 3.0 / batchsize

        real_loss += F.sum(F.softplus(-y1_real)) / 3.0 / batchsize
        real_loss += F.sum(F.softplus(-y2_real)) / 3.0 / batchsize
        real_loss += F.sum(F.softplus(-y3_real)) / 3.0 / batchsize

        loss = fake_loss + real_loss
        chainer.report({'loss': loss}, dis)
        return loss

    def update_core(self):
        enc_optimizer = self.get_optimizer('enc')
        dec_optimizer = self.get_optimizer('dec')
        dis_optimizer = self.get_optimizer('dis')

        enc, dec, dis = self.enc, self.dec, self.dis
        xp = enc.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        batch_seq = chainer.dataset.convert.concat_examples(batch, self.device)
        x1 = chainer.Variable(batch_seq[:, 0])
        x2 = chainer.Variable(batch_seq[:, 1])
        x3 = chainer.Variable(batch_seq[:, 2])

        # encode x -> mu, sig
        mu1, ln_var1 = enc(x1)
        mu3, ln_var3 = enc(x3)
        # Monte Carlo Sanpling mu, sig -> z l = 1
        z1 = F.gaussian(mu1, ln_var1)
        z3 = F.gaussian(mu3, ln_var3)
        z2 = (z1 + z3) / 2
        # decode z -> x
        x1_rec = dec(z1)
        x2_rec = dec(z2)
        x3_rec = dec(z3)
        # disctiminate x -> y
        y1_fake = F.sigmoid(dis(x1_rec))
        y2_fake = F.sigmoid(dis(x2_rec))
        y3_fake = F.sigmoid(dis(x3_rec))
        y1_real = dis(x1)
        y2_real = dis(x2)
        y3_real = dis(x3)

        enc_optimizer.update(self.enc_loss, enc, x1, x2, x3, mu1, mu3, ln_var1, ln_var3, x1_rec, x2_rec, x3_rec, y1_fake, y2_fake, y3_fake)
        dec_optimizer.update(self.dec_loss, dec, x1, x2, x3, x1_rec, x2_rec, x3_rec, y1_fake, y2_fake, y3_fake)
        dis_optimizer.update(self.dis_loss, dis, y1_fake, y2_fake, y3_fake, y1_real, y2_real, y3_real)
