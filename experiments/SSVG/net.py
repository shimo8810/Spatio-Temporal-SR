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

class Encoder(chainer.Chain):
    def __init__(self, size, ch, n_latent, ch_scale):
        super(Encoder, self).__init__()
        self.ch = ch # データのチャネル数
        self.size = size # データの縦横サイズ(同じ以外のデータを食わせる予定はない)
        self.ch_scale = ch_scale # チャネル数のスケール
        self.n_latent = n_latent # 潜在変数次元
        self.z_ch = ch_scale * (2**4) # 潜在変数に変換する直前のMAPのチャネル数
        self.z_size = size // (2**4) # 潜在変数に変換する直前のMAPのサイズ
        self.n_flat = self.z_ch * (self.z_size**2) # 潜在変数に変換する前のMAPをフラットにしたサイズ
        init_w = chainer.initializers.HeNormal()

        with self.init_scope():
            self.conv0 = L.Convolution2D(None, ch_scale, 3, 1, 1, initialW=init_w)
            self.conv1 = ConvBNR(ch_scale * 1, ch_scale *  2)# 32-64 -> 64-32
            self.conv2 = ConvBNR(ch_scale * 2, ch_scale *  4)# 64-32 -> 128-16
            self.conv3 = ConvBNR(ch_scale * 4, ch_scale *  8)# 128-16 -> 256-8
            self.conv4 = ConvBNR(ch_scale * 8, ch_scale * 16)# 256-8 -> 512-4
            self.fc5_mu = L.Linear(self.n_flat, n_latent)
            self.fc5_ln_var = L.Linear(self.n_flat, n_latent)

    def forward(self, x):
        h = F.leaky_relu(self.conv0(x))
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        print("flatshape:", h.shape, self.z_size, self.z_ch)
        mu = self.fc5_mu(h)
        ln_var = self.fc5_ln_var(h)
        return mu, ln_var

class Decoder(chainer.Chain):
    def __init__(self,  size, ch, n_latent, ch_scale):
        super(Decoder, self).__init__()
        self.ch = ch # データのチャネル数
        self.size = size # データの縦横サイズ(同じ以外のデータを食わせる予定はない)
        self.ch_scale = ch_scale # チャネル数のスケール
        self.n_latent = n_latent # 潜在変数次元
        self.z_ch = ch_scale * (2**4) # 潜在変数に変換する直前のMAPのチャネル数
        self.z_size = size // (2**4) # 潜在変数に変換する直前のMAPのサイズ
        self.n_flat = self.z_ch * (self.z_size**2) # 潜在変数に変換する前のMAPをフラットにしたサイズ

        init_w = chainer.initializers.HeNormal()

        with self.init_scope():
            self.fc0 = L.Linear(n_latent, self.n_flat)
            self.conv1 = ConvBNR(ch_scale * 16, ch_scale * 8, sample="up")
            self.conv2 = ConvBNR(ch_scale *  8, ch_scale * 4, sample="up")
            self.conv3 = ConvBNR(ch_scale *  4, ch_scale * 2, sample="up")
            self.conv4 = ConvBNR(ch_scale *  2, ch_scale * 1, sample="up")
            self.conv5 = L.Convolution2D(ch_scale, self.ch, 3, 1, 1, initialW=init_w)

    def forward(self, z):
        h = F.tanh(self.fc0(z))
        h = F.reshape(h, (z.shape[0], self.z_ch, self.z_size, self.z_size))
        print("dec flat shape:", h.shape)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        return h

class Discriminator(chainer.Chain):
    def __init__(self, ch_scale):
        super(Discriminator, self).__init__()
        init_w = chainer.initializers.HeNormal()
        ch_scale = 8

        with self.init_scope():
            self.conv0 = L.Convolution2D(None, ch_scale, 3, 1, 1, initialW=init_w)
            self.conv1 = ConvBNR(ch_scale * 1, ch_scale * 2)
            self.conv2 = ConvBNR(ch_scale * 2, ch_scale * 4)
            self.conv3 = ConvBNR(ch_scale * 4, ch_scale * 8)
            self.conv4 = ConvBNR(ch_scale * 8, ch_scale *16)
            self.conv5 = L.Convolution2D(None, 1, 3, 1, 1, initialW=init_w)

    def forward(self, x):
        h = F.leaky_relu(self.conv0(x))
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        return h
