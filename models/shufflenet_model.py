"""
shufflenet_model.py

Lightweight ShuffleNetV2-like model implemented in tensorflow.keras.
Designed to accept 48x48 grayscale face crops (1 channel) and output softmax
probabilities for N emotion classes (e.g., ['angry','fatigue','drowsy','neutral']).

Provides:
- build_shufflenetv2(input_shape=(48,48,1), num_classes=4, width_multiplier=1.0)
- load_emotion_weights(model, weights_path)
- Example of model.summary() usage (commented).
"""

from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, BatchNormalization,
    ReLU, Add, GlobalAveragePooling2D, Dense
)
from tensorflow.keras.models import Model
import tensorflow as tf
import os

def channel_shuffle(x, groups):
    """
    Channel shuffle for ShuffleNet.
    x: tensor with shape (batch, h, w, c)
    groups: number of groups to split channels into
    """
    # shape: (batch, h, w, groups, channels_per_group)
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    channels = x.shape[-1]
    if channels is None:
        raise ValueError("Number of channels must be known (not None).")
    channels_per_group = channels // groups
    # reshape and transpose to shuffle channels
    x = tf.reshape(x, [-1, height, width, groups, channels_per_group])
    x = tf.transpose(x, perm=[0,1,2,4,3])  # swap group and channels_per_group
    x = tf.reshape(x, [-1, height, width, channels])
    return x

def conv_bn_relu(x, filters, kernel_size=1, strides=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def depthwise_conv_bn(x, kernel_size=3, strides=1):
    x = DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    return x

def shufflenet_unit(x, out_channels, strides=1, groups=2):
    """
    A simplified ShuffleNetV2 unit.
    - If strides==1: channels are split into two branches (identity + transform).
    - If strides==2: both branches transform and concatenate.
    This is a compact variant suitable for small input sizes.
    """
    in_channels = x.shape[-1]
    if strides == 1:
        # channel split
        assert in_channels % 2 == 0, "in_channels must be divisible by 2 for split"
        split = in_channels // 2
        x1 = layers.Lambda(lambda z: z[:, :, :, :split])(x)  # identity branch
        x2 = layers.Lambda(lambda z: z[:, :, :, split:])(x)  # transform branch

        # transform branch
        y = conv_bn_relu(x2, out_channels // 2, kernel_size=1)
        y = depthwise_conv_bn(y, kernel_size=3)
        y = BatchNormalization()(y)
        y = conv_bn_relu(y, out_channels // 2, kernel_size=1)
        # concat
        out = layers.Concatenate(axis=-1)([x1, y])
        out = layers.Lambda(lambda z: channel_shuffle(z, groups))(out)
        return out
    else:
        # strides == 2 branch (downsample)
        # branch 1
        y1 = depthwise_conv_bn(x, kernel_size=3, strides=2)
        y1 = conv_bn_relu(y1, out_channels // 2, kernel_size=1)

        # branch 2
        y2 = conv_bn_relu(x, out_channels // 2, kernel_size=1)
        y2 = depthwise_conv_bn(y2, kernel_size=3, strides=2)
        y2 = conv_bn_relu(y2, out_channels // 2, kernel_size=1)

        out = layers.Concatenate(axis=-1)([y1, y2])
        out = layers.Lambda(lambda z: channel_shuffle(z, groups))(out)
        return out

def build_shufflenetv2(input_shape=(48,48,1), num_classes=4, width_multiplier=1.0):
    """
    Build a compact ShuffleNetV2-like architecture.
    - input_shape: (48,48,1) grayscale face crop
    - num_classes: number of emotion classes
    - width_multiplier: scale channels (e.g., 0.5, 1.0, 1.5)
    """
    if len(input_shape) != 3:
        raise ValueError("input_shape must be tuple (H,W,C)")
    inp = Input(shape=input_shape)

    # Stem: small conv
    x = conv_bn_relu(inp, int(24 * width_multiplier), kernel_size=3, strides=2)  # 24 ch
    x = depthwise_conv_bn(x, kernel_size=3)  # keep size small

    # stages: use small sequence because input is tiny (48x48) and dataset is small
    out_channels_list = [
        int(48 * width_multiplier),
        int(96 * width_multiplier),
        int(192 * width_multiplier)
    ]

    # Stage 1 (downsample)
    x = shufflenet_unit(x, out_channels_list[0], strides=2)
    for i in range(1):  # repeat a small number of units
        x = shufflenet_unit(x, out_channels_list[0], strides=1)

    # Stage 2
    x = shufflenet_unit(x, out_channels_list[1], strides=2)
    for i in range(2):
        x = shufflenet_unit(x, out_channels_list[1], strides=1)

    # Stage 3
    x = shufflenet_unit(x, out_channels_list[2], strides=2)
    for i in range(3):
        x = shufflenet_unit(x, out_channels_list[2], strides=1)

    # Head
    x = GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out, name="ShuffleNetV2_small")
    return model

def load_emotion_weights(model, weights_path):
    """
    Loads pretrained weights into the model if available.
    Returns True if loaded successfully, False otherwise.
    """
    if weights_path is None:
        return False
    if not os.path.exists(weights_path):
        print(f"[WARN] Weights file not found: {weights_path}")
        return False
    try:
        model.load_weights(weights_path)
        print(f"[INFO] Loaded weights from {weights_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        return False

# If module run directly, build a small model and print summary (useful during dev)
if __name__ == "__main__":
    model = build_shufflenetv2((48,48,1), num_classes=4, width_multiplier=0.5)
    model.summary()
