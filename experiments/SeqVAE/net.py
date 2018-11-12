"""
ネットワーク
"""
import chainer
import chainer.functions as F
import chainer.links as L

class ConvBNR(chainer.Chain):
    """
    Convolution -> (BatchNormalization) -> (Dropout) -> Activation
    """
    def __init__(self, ch_in, ch_out, sample="down", use_bn=True, activation=F.leaky_relu, use_dp=False):
        self.use_bn = use_bn
        self.activation = activation
        self.use_dp = use_dp

        init_w = chainer.initializers.HeNormal()

        super(ConvBNR, self).__init__()

        with self.init_scope():
            if sample == "down":
                self.conv = L.Convolution2D(
                    ch_in, ch_out, ksize=4, stride=2, pad=1, initialW=init_w)
            elif sample == "up":
                self.conv = L.Deconvolution2D(
                    ch_in, ch_out, ksize=4, stride=2, pad=1, initialW=init_w)
            else:
                raise ValueError('argument "sample" must be "down" or "up".')
            if self.use_bn:
                self.bn = L.BatchNormalization(ch_out)

    def forward(self, x):
        h = self.conv(x)
        if self.use_bn:
            h = self.bn(h)
        if self.use_dp:
            h = F.dropout(h)
        if self.activation is not None:
            h = self.activation(h)
        return h

class SeqVAE(chainer.Chain):
    def __init__(self, size, ch, n_latent, ch_scale, activation=F.relu):
        self.ch = ch # データのチャネル数
        self.size = size # データの縦横サイズ(同じ以外のデータを食わせる予定はない)
        self.ch_scale = ch_scale # チャネル数のスケール
        self.n_latent = n_latent # 潜在変数次元
        self.z_ch = ch_scale * (2**4) # 潜在変数に変換する直前のMAPのチャネル数
        self.z_size = size // (2**4) # 潜在変数に変換する直前のMAPのサイズ
        self.n_flat = self.z_ch * (self.z_size**2) # 潜在変数に変換する前のMAPをフラットにしたサイズ

        init_w = chainer.initializers.HeNormal()

        super(SeqVAE, self).__init__()

        with self.init_scope():
            # encoder 128 -> 128 -> 64 -> 32 -> 16 -> 8
            self.enc_conv0 = L.Convolution2D(None, ch_scale, 3, 1, 1, initialW=init_w)
            self.enc_conv1 = ConvBNR(ch_scale * 1, ch_scale *  2, activation=activation)
            self.enc_conv2 = ConvBNR(ch_scale * 2, ch_scale *  4, activation=activation)
            self.enc_conv3 = ConvBNR(ch_scale * 4, ch_scale *  8, activation=activation)
            self.enc_conv4 = ConvBNR(ch_scale * 8, ch_scale * 16, activation=activation)
            self.enc_fc5_mu = L.Linear(self.n_flat, n_latent)
            self.enc_fc5_ln_var = L.Linear(self.n_flat, n_latent)

            # decoder
            self.dec_fc0 = L.Linear(n_latent, self.n_flat)
            self.dec_conv1 = ConvBNR(ch_scale * 16, ch_scale * 8, sample="up")
            self.dec_conv2 = ConvBNR(ch_scale *  8, ch_scale * 4, sample="up")
            self.dec_conv3 = ConvBNR(ch_scale *  4, ch_scale * 2, sample="up")
            self.dec_conv4 = ConvBNR(ch_scale *  2, ch_scale * 1, sample="up")
            self.dec_conv5 = L.Convolution2D(ch_scale, self.ch, 3, 1, 1, initialW=init_w)

    def encode(self, x):
        h = F.leaky_relu(self.enc_conv0(x))
        h = self.enc_conv1(h)
        h = self.enc_conv2(h)
        h = self.enc_conv3(h)
        h = self.enc_conv4(h)
        mu = self.enc_fc5_mu(h)
        ln_var = self.enc_fc5_ln_var(h)
        return mu, ln_var

    def decode(self, z, sig=True):
        h = F.tanh(self.dec_fc0(z))
        h = F.reshape(h, (z.shape[0], self.z_ch, self.z_size, self.z_size))
        h = self.dec_conv1(h)
        h = self.dec_conv2(h)
        h = self.dec_conv3(h)
        h = self.dec_conv4(h)
        h = self.dec_conv5(h)
        if sig:
            return F.sigmoid(h)
        else:
            return h

    def forward(self, x, sig=True):
        return self.decode(self.encode(x)[0], sig)

    def get_loss_func(self, C=1.0, k=1):
        def lf(x):
            mu, ln_var = self.encode(x)
            batch_size = len(mu.data)

            # reconstruction loss
            rec_loss = 0
            for l in range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decode(z, sig=False)) / (k * batch_size)
            self.rec_loss = rec_loss

            # kl-divergnece regularization
            self.kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batch_size
            self.loss = self.rec_loss + C * self.kl_loss
            chainer.report({'rec_loss': self.rec_loss,
                'kl_loss':self.kl_loss, 'loss': self.loss}, observer=self)
            return self.loss
        return lf

    def get_seq_loss_func(self, C1=1.0, C2=1.0, k=1):
        def lf(x):
            bachsize, numframe, _, _, _ = x.shape
            x1 = x[:,0,:]
            x3 = x[:,2,:]
            mu1, ln_var1 = self.encode(x1)
            mu3, ln_var3 = self.encode(x3)

            # reconstruction loss
            rec_loss = 0
            seq_loss = 0
            for l in range(k):
                z1 = F.gaussian(mu1, ln_var1)
                rec_loss += F.bernoulli_nll(x1, self.decode(z1, sig=False)) / (2 * k * bachsize)
                z3 = F.gaussian(mu3, ln_var3)
                rec_loss += F.bernoulli_nll(x3, self.decode(z3, sig=False)) / (2 * k * bachsize)

                z2 = (z1 + z3) / 2
                seq_loss += F.bernoulli_nll(x[:,1,:], self.decode(z2, sig=False)) / (k * bachsize)
            self.rec_loss = rec_loss
            self.seq_loss = seq_loss

            # kl-divergnece regularization
            self.kl_loss = F.gaussian_kl_divergence(mu1, ln_var1) / bachsize / 2.0
            self.kl_loss += F.gaussian_kl_divergence(mu3, ln_var3) / bachsize / 2.0

            # loss summation
            self.loss = self.rec_loss + C1 * self.kl_loss + C2 * self.seq_loss
            chainer.report({'rec_loss': self.rec_loss,
                'kl_loss':self.kl_loss, 'seq_loss': self.seq_loss, 'loss': self.loss}, observer=self)
            return self.loss
        return lf
