import tensorflow.keras as K

L = K.layers


def ELUCons(elu):
    if elu:
        return L.ELU()
    else:
        return L.PReLU(shared_axes=[1, 2])


class LUConv(L.Layer):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.nchan = nchan
        self.elu = elu
        self.relu1 = ELUCons(elu)
        self.conv1 = L.Conv3D(nchan, kernel_size=5, padding="same")
        self.bn1 = L.BatchNormalization(momentum=0.9, epsilon=1e-5)

    def call(self, x, **kwargs):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

    def get_config(self):
        config = super(LUConv, self).get_config()
        config.update({'nchan': self.nchan, 'elu': self.elu})
        return config

def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return K.Sequential(layers)


class InputTransition(L.Layer):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.outChans = outChans
        self.elu = elu

        self.conv1 = L.Conv3D(outChans, kernel_size=5, padding="same")
        self.bn1 = L.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu1 = ELUCons(elu)
        self.add = L.Add()

    def call(self, x, **kwargs):
        out = self.bn1(self.conv1(x))
        out = self.relu1(self.add([out, x]))    # auto broadcast
        return out

    def get_config(self):
        config = super(InputTransition, self).get_config()
        config.update({'outChans': self.outChans, 'elu': self.elu})
        return config


class DownTransition(L.Layer):
    def __init__(self, outChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        self.outChans = outChans
        self.nConvs = nConvs
        self.elu = elu
        self.dropout = dropout

        self.down_conv = L.Conv3D(outChans, kernel_size=2, strides=2)
        self.bn1 = L.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu1 = ELUCons(elu)
        self.relu2 = ELUCons(elu)
        self.do1 = lambda x: x
        if dropout:
            self.do1 = L.Dropout(rate=0.5)
        self.ops = _make_nConv(outChans, nConvs, elu)
        self.add = L.Add()

    def call(self, x, **kwargs):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(self.add([out, down]))
        return out

    def get_config(self):
        config = super(DownTransition, self).get_config()
        config.update({'outChans': self.outChans, 'nConvs': self.nConvs,
                       'elu': self.elu, 'dropout': self.dropout})
        return config


class UpTransition(L.Layer):
    def __init__(self, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.outChans = outChans
        self.nConvs = nConvs
        self.elu = elu
        self.dropout = dropout

        self.up_conv = L.Conv3DTranspose(outChans // 2, kernel_size=2, strides=2)
        self.bn1 = L.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.do1 = lambda x: x
        self.do2 = L.Dropout(rate=0.5)
        self.relu1 = ELUCons(elu)
        self.relu2 = ELUCons(elu)
        if dropout:
            self.do1 = L.Dropout(rate=0.5)
        self.ops = _make_nConv(outChans, nConvs, elu)
        self.add = L.Add()
        self.concat = L.Concatenate()

    def call(self, x, **kwargs):
        x, skipx = x
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = self.concat([out, skipxdo])
        out = self.ops(xcat)
        out = self.relu2(self.add([out, xcat]))
        return out

    def get_config(self):
        config = super(UpTransition, self).get_config()
        config.update({'outChans': self.outChans, 'nConvs': self.nConvs,
                       'elu': self.elu, 'dropout': self.dropout})
        return config


class OutputTransition(L.Layer):
    def __init__(self, ncls, elu):
        super(OutputTransition, self).__init__()
        self.ncls = ncls
        self.elu = elu

        self.conv1 = L.Conv3D(ncls, kernel_size=5, padding="same")
        self.bn1 = L.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = L.Conv3D(ncls, kernel_size=1)
        self.relu1 = ELUCons(elu)
        self.softmax = L.Softmax()

    def call(self, x, **kwargs):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # out = self.softmax(out)   # Don't use.
        return out

    def get_config(self):
        config = super(OutputTransition, self).get_config()
        config.update({'ncls': self.ncls, 'elu': self.elu})
        return config


def VNet(shape, batch_size, ncls=2, elu=True, name="vnet"):
    x = K.Input(shape=shape, batch_size=batch_size)
    out16 = InputTransition(16, elu)(x)
    out32 = DownTransition(32, 1, elu)(out16)
    out64 = DownTransition(64, 2, elu)(out32)
    out128 = DownTransition(128, 3, elu, dropout=True)(out64)
    out256 = DownTransition(256, 2, elu, dropout=True)(out128)
    out = UpTransition(256, 2, elu, dropout=True)([out256, out128])
    out = UpTransition(128, 2, elu, dropout=True)([out, out64])
    out = UpTransition(64, 1, elu)([out, out32])
    out = UpTransition(32, 1, elu)([out, out16])
    y = OutputTransition(ncls, elu)(out)
    return K.Model(inputs=x, outputs=y, name=name)
