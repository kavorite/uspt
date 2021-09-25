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
        x = tf.keras.layers.LayerNormalization()(x)
        spatial_embedding = x
        for layer in (
            tf.keras.layers.SeparableConv2D(
                x.shape[-1],
                x.shape[-3:-1],
                strides=1,
                use_bias=False,
                kernel_initializer="glorot_normal",
            ),
            lambda x: squeeze_excitation(x, 1),
            tf.keras.layers.Flatten(),
        ):
            spatial_embedding = layer(spatial_embedding)
        x = (
            spatial_embedding
            + tf.keras.layers.GlobalAveragePooling2D()(squeeze_excitation(x, 1))
            + tf.keras.layers.GlobalMaxPooling2D()(squeeze_excitation(x, 1))
        )
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.LayerNormalization(name="xfeatures")(x)
    hsvds = tf.keras.layers.Dense(3, name="hsv_offset")(x)
    theta = tf.keras.layers.Dense(1, name="rot_factor")(x)
    delta = tf.keras.layers.Dense(2, name="tsl_offset")(x)
    alpha = tf.keras.layers.Dense(1, name="scl_factor")(x)
    output = dict(
        hsv_offset=hsvds, rot_factor=theta, tsl_offset=delta, scl_factor=alpha
    )
    return tf.keras.Model(
        surrogate.input, output, name=surrogate.name + "_unsupervised"
    )


def model_stem(model):
    return model.get_layer("xfeatures")
