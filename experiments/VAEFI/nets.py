"""
network model
"""
import chainer
import chainer.links as L
import chainer.functions as F

class ConvBNR(chainer.Chain):
    """
    Convolution -> (BatchNormalization) -> (Dropout) -> Activation
    """
    def __init__(self, ch_in, ch_out, sample="down", use_bn=True, activation=F.relu, use_dp=False):
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

class CNNVAE(chainer.Chain):
    def __init__(self, size, n_latent, n_ch, activation=F.relu):
        self.n_ch = n_ch
        self.n_latent = n_latent
        self.size = size
        # self.z_size = (n_ch * (2^4)) * (size // (2^4)) ^ 2
        self.z_ch = n_ch * (2**4)
        self.z_size = size // (2**4)
        self.z_dim = self.z_ch * (self.z_size**2)

        init_w = chainer.initializers.HeNormal()

        super(CNNVAE, self).__init__()

        with self.init_scope():
            # encoder 128 -> 128 -> 64 -> 32 -> 16 -> 8
            self.enc_conv0 = L.Convolution2D(None, n_ch, 3, 1, 1, initialW=init_w)
            self.enc_conv1 = ConvBNR(n_ch * 1, n_ch * 2, activation=activation)
            self.enc_conv2 = ConvBNR(n_ch * 2, n_ch * 4, activation=activation)
            self.enc_conv3 = ConvBNR(n_ch * 4, n_ch * 8, activation=activation)
            self.enc_conv4 = ConvBNR(n_ch * 8, n_ch * 16, activation=activation)
            self.enc_fc5 = L.Linear(self.z_dim, 2000)
            self.enc_fc6_mu = L.Linear(2000, n_latent)
            self.enc_fc6_ln_var = L.Linear(2000, n_latent)

            # decoder
            self.dec_fc0 = L.Linear(n_latent, self.z_dim)
            self.dec_conv1 = ConvBNR(n_ch *16, n_ch * 8, sample="up")
            self.dec_conv2 = ConvBNR(n_ch * 8, n_ch * 4, sample="up")
            self.dec_conv3 = ConvBNR(n_ch * 4, n_ch * 2, sample="up")
            self.dec_conv4 = ConvBNR(n_ch * 2, n_ch * 1, sample="up")
            self.dec_conv5 = L.Convolution2D(n_ch, 3, 3, 1, 1, initialW=init_w)

    def encode(self, x):
        h = F.leaky_relu(self.enc_conv0(x))
        h = self.enc_conv1(h)
        h = self.enc_conv2(h)
        h = self.enc_conv3(h)
        h = self.enc_conv4(h)
        h = self.enc_fc5(h)
        mu = self.enc_fc6_mu(h)
        ln_var = self.enc_fc6_ln_var(h)
        return mu, ln_var

    def decode(self, z, sig=True):
        h = self.dec_fc0(z)
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
            chainer.report(
                {'rec_loss': self.rec_loss, 'kl_loss':self.kl_loss, 'loss': self.loss}, observer=self)
            return self.loss
        return lf
