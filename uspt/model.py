import tensorflow as tf


def squeeze_excitation(x, channels):
    squeezed = x
    for layer in (
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(channels),
        tf.keras.layers.Activation(tf.nn.silu),
        tf.keras.layers.Dense(channels),
        tf.keras.layers.Activation(tf.nn.sigmoid),
    ):
        squeezed = layer(x)
    return tf.keras.layers.Multiply()([x, squeezed])


def base_model(image_shape):
    return tf.keras.applications.DenseNet121(
        input_shape=image_shape, weights=None, include_top=False
    )


def build_model(image_shape, surrogate=None):
    if surrogate is None:
        surrogate = base_model(image_shape)
    x = surrogate.output
    dummy_input = tf.random.uniform([1, *image_shape])
    if tf.rank(tf.squeeze(surrogate.predict(dummy_input))) != 1:
        # learned, volumetric downsampling
        x = squeeze_excitation(x, 1)
        x = tf.keras.layers.SeparableConv2D(x.shape[-1], x.shape[-3:-1], strides=1)(x)
    x = tf.keras.layers.Flatten()(x)
    lns = iter(tf.keras.layers.LayerNormalization() for _ in range(3))
    hsvds = tf.keras.layers.Dense(3, name="hsv_offset")(next(lns)(x))
    theta = tf.keras.layers.Dense(1, name="rot_factor")(next(lns)(x))
    delta = tf.keras.layers.Dense(1, name="tsl_offset")(next(lns)(x))
    output = dict(hsv_offset=hsvds, rot_factor=theta, tsl_offset=delta)
    return tf.keras.Model(
        surrogate.input, output, name=surrogate.name + "_unsupervised"
    )
