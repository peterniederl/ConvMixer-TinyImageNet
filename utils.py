from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa


class GroupConv2D(layers.Layer):
    def __init__(self, input_channels, output_channels, kernel_size=(3,3), padding='same', groups=1, strides=1, kernel_initializer="glorot_uniform", use_bias=True, **kwargs):
        super().__init__(**kwargs)
        assert input_channels % groups == 0, "in_ch must be divisible by groups"
        assert output_channels % groups == 0, "out_ch must be divisible by groups"
        self.in_ch = input_channels
        self.out_ch = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.convs = []
        self.strides = strides
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias

    def build(self, input_shape):
        for _ in range(self.groups):
            self.convs.append(
                layers.Conv2D(self.out_ch // self.groups, self.kernel_size, padding=self.padding, strides=self.strides, kernel_initializer=self.kernel_initializer, use_bias=self.use_bias)
            )

    def call(self, x):
        splits = tf.split(x, num_or_size_splits=self.groups, axis=-1)
        outs = [conv(s) for conv, s in zip(self.convs, splits)]
        return tf.concat(outs, axis=-1)
    
    
class DropPath(layers.Layer):
    """Stochastic Depth per sample (ConvNet-friendly)"""

    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)

    def call(self, x, training=None):
        if (not training) or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob

        # Shape: (B, 1, 1, 1) for ConvMixer
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)

        # IMPORTANT: cast everything to x.dtype
        random_tensor = keep_prob + tf.random.uniform(
            shape,
            dtype=x.dtype
        )

        binary_tensor = tf.floor(random_tensor)
        return tf.divide(x, keep_prob) * binary_tensor
    


class SEBlock(layers.Layer):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(
            channels // reduction,
            activation="gelu",
            use_bias=True
        )
        self.fc2 = layers.Dense(
            channels,
            activation="sigmoid",
            use_bias=True
        )

    def call(self, x):
        s = self.pool(x)
        s = self.fc1(s)
        s = self.fc2(s)
        s = tf.reshape(s, [-1, 1, 1, x.shape[-1]])
        return x * s
    

def spatial_shuffle(x):
    # x: [B, H, W, C], H and W even
    B, H, W, C = tf.unstack(tf.shape(x))

    x = tf.reshape(x, [B, H // 2, 2, W // 2, 2, C])
    # dimensions:
    # [row_block, row_parity, col_block, col_parity]

    # move parities into spatial layout
    x = tf.transpose(x, [0, 2, 4, 1, 3, 5])
    # [row_parity, col_parity, row_block, col_block]

    return tf.reshape(x, [B, H, W, C])


def channel_shuffle(x, groups):
    B, H, W, C = tf.unstack(tf.shape(x))

    tf.debugging.assert_equal(
        C % groups, 0,
        message="Channels must be divisible by groups"
    )

    x = tf.reshape(x, [B, H, W, groups, C // groups])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    return tf.reshape(x, [B, H, W, C])


class SpatialShuffle(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        x_shuffle = spatial_shuffle(x)
        x = tf.concat([x, x_shuffle], axis=-1)
        x = channel_shuffle(x, 2)
        return x


class ConvMixer(layers.Layer):
    """
    Mode-based Vision Mixer block.

    Layout: [B, H, W, C]

    Modes:
      - "local":   Conv token mixing + channel mixing
      - "inner":   Per-patch dense token mixing + channel mixing
      - "global":  Per-patch dense + global token MLP + channel mixing
    """

    def __init__(
        self,
        dim,
        attr={'kernel_size': 3},
        mode='depthwise',
        use_se=False,
    ):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.use_se = use_se
        self.drop_path = DropPath(attr.get('drop_rate', 0))

        if mode == 'depthwise':
            self.dw = tf.keras.Sequential([
                layers.DepthwiseConv2D(kernel_size=attr['kernel_size'], padding='same', use_bias=False),
                layers.Activation("gelu"),
                layers.BatchNormalization()
            ])    
            
        if mode == 'axial_depthwise':
            self.dw = tf.keras.Sequential([
                layers.DepthwiseConv2D(kernel_size=(1, attr['axial_kernel_size']), padding='same', use_bias=False),
                layers.DepthwiseConv2D(kernel_size=(attr['axial_kernel_size'], 1), padding='same', use_bias=False),
                layers.Activation("gelu"),
                layers.BatchNormalization()
            ]) 
            
        if mode == 'mixed':
            self.dw = tf.keras.Sequential([
                layers.DepthwiseConv2D(kernel_size=(1, attr['axial_kernel_size']), padding='same', use_bias=False),
                layers.DepthwiseConv2D(kernel_size=(attr['axial_kernel_size'], 1), padding='same', use_bias=False),
                layers.Activation("gelu"),
                layers.DepthwiseConv2D(kernel_size=attr['kernel_size'], padding='same', use_bias=False),
                layers.BatchNormalization()
            ]) 

        self.gamma = self.add_weight(
            shape=(dim,),
            initializer="ones",
            trainable=True
        )
                    

        self.pw = tf.keras.Sequential([
            layers.Conv2D(filters=dim, kernel_size=1, padding='same', use_bias=False),
            layers.Activation("gelu"),
            layers.BatchNormalization()
        ])      
        if self.use_se:
            self.se_block = SEBlock(dim, reduction=8)
        

    def call(self, x, training=None):
        y = self.dw(x)
        if self.use_se:
            y = self.se_block(y)
        x = x + self.drop_path(y * self.gamma, training) 
        x = self.pw(x)

        return x
