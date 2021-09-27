import tensorflow as tf

from .data import make_xform_annotator


def se_block(x, channels):
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


def build_encoder(image_shape, augment_difficulty=0.0):
    inputs = tf.keras.layers.Input(image_shape)
    if augment_difficulty > 0:
        for augmentation in (
            tf.keras.layers.RandomContrast(augment_difficulty),
            tf.keras.layers.RandomZoom(augment_difficulty, augment_difficulty),
            tf.keras.layers.RandomFlip("horizontal"),
        ):
            inputs = augmentation(inputs)
    encoder = tf.keras.applications.DenseNet121(
        input_shape=image_shape, weights=None, include_top=False
    )
    outputs = encoder(inputs)
    dummy_input = tf.zeros(encoder.input.shape[1:])[None, ...]
    dummy_output = encoder(dummy_input, training=False)
    if tf.rank(tf.squeeze(dummy_output)) != 1:
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(se_block(outputs, 1))
        max_pool = tf.keras.layers.GlobalMaxPooling2D()(se_block(outputs, 1))
        outputs = tf.keras.layers.Add(name="encoding")([avg_pool, max_pool])
    return tf.keras.Model(inputs, outputs, name="uspt_encoder")


def add_projection_head(encoder, project_dim):
    prefix = encoder.name
    projection = encoder.get_layer("encoding").output
    for layer in (
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(project_dim, name="projection"),
        tf.keras.layers.Activation(tf.nn.silu),
    ):
        projection = layer(projection)
    return tf.keras.Model(encoder.input, projection, name=f"{prefix}_projector")


def add_xform_heads(encoder):
    x = encoder.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.LayerNormalization()(x)
    hsvds = tf.keras.layers.Dense(3, name="hsv_offset")(x)
    theta = tf.keras.layers.Dense(1, name="rot_factor")(x)
    delta = tf.keras.layers.Dense(2, name="tsl_offset")(x)
    alpha = tf.keras.layers.Dense(1, name="scl_factor")(x)
    output = dict(
        hsv_offset=hsvds, rot_factor=theta, tsl_offset=delta, scl_factor=alpha
    )
    return tf.keras.Model(encoder.input, output, name=encoder.name + "_uspt_xform")


def build_predictor(project_dim, latent_dim, weight_decay):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[project_dim]),
            tf.keras.layers.Dense(
                latent_dim,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            ),
            tf.keras.layers.Activation(tf.nn.silu),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(project_dim),
        ],
        name="predictor",
    )


class SimSiam(tf.keras.Model):
    def __init__(
        self,
        projector=add_projection_head(
            build_encoder(image_shape=[512, 512, 3]), project_dim=256
        ),
        predictor=build_predictor(project_dim=256, latent_dim=128, weight_decay=0.0),
        xformer=make_xform_annotator(include_xforms=False),
    ):
        """
        Constructs a SimSiam (simple siamese) unsupervised training harness.

        Parameters
        -------
        projector: Image feature extraction backbone with projection head
        attached.
        predictor: Autoencoder for projected representations.
        xformer: Data augmentation function for images, or None, if the model
        includes preprocessing layers that perform data augmentation.

        References
        -------
        https://keras.io/examples/vision/simsiam/
        """
        super().__init__(self)
        self.projector = projector
        self.predictor = predictor
        self.xformer = xformer if xformer is not None else (lambda x: x)
        self.loss_tr = tf.keras.metrics.Mean(name="cos_dissimilarity")
        self.build(projector.input.shape)

    @staticmethod
    def cos_dissimilarity(p, z):
        """
        Compute stop-gradient Simple Siamese objective between two latent
        representations.
        """
        z = tf.stop_gradient(z)
        p = tf.math.l2_normalize(p, axis=-1)
        z = tf.math.l2_normalize(z, axis=-1)
        return 1 - tf.reduce_mean(tf.reduce_sum(p * z), axis=-1)

    @property
    def metrics(self):
        return [self.loss_tr]

    def train_step(self, image):
        x, y = self.xformer(image), self.xformer(image)
        with tf.GradientTape() as tape:
            p = self.projector(x), self.projector(y)
            z = self.predictor(p[0]), self.predictor(p[1])
            loss = 0.5 * self.cos_dissimilarity(
                p[0], z[1]
            ) + 0.5 * self.cos_dissimilarity(p[1], z[0])
        train = self.projector.trainable_variables + self.predictor.trainable_variables
        grads = tape.gradient(loss, train)
        self.optimizer.apply_gradients(zip(grads, train))

    def call(self, image, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        return self.projector(image, training=training)
