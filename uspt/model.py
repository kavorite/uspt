import tensorflow as tf


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


def build_augmenter(difficulty=0.10):
    layers = [
        tf.keras.layers.RandomContrast(difficulty),
        tf.keras.layers.RandomZoom(difficulty, difficulty),
        tf.keras.layers.RandomFlip("horizontal"),
    ]
    return tf.keras.Sequential(layers, name="data_augmentation")


def build_encoder(
    backbone=tf.keras.applications.Xception(
        input_shape=(224, 224, 3), weights=None, include_top=False
    ),
    augmenter=build_augmenter(),
):
    inputs = tf.keras.layers.Input(backbone.input.shape[1:])
    if augmenter is not None:
        outputs = augmenter(inputs)
    outputs = backbone(inputs)
    max_pool = tf.keras.layers.GlobalMaxPool2D()(se_block(outputs, 1))
    avg_pool = tf.keras.layers.GlobalAvgPool2D()(se_block(outputs, 1))
    outputs = tf.keras.layers.Add(name="encoding")([max_pool, avg_pool])
    return tf.keras.Model(inputs, outputs, name="uspt_encoder")


def add_projection_head(encoder, project_dim):
    prefix = encoder.name
    projection = encoder.get_layer("encoding").output
    for layer in (
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(project_dim, name="projection"),
        tf.keras.layers.Activation(tf.nn.silu),
    ):
        projection = layer(projection)
    return tf.keras.Model(encoder.input, projection, name=f"{prefix}_projector")


def add_xform_heads(encoder):
    x = encoder.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    hsvds = tf.keras.layers.Dense(3, name="hsv_offset")(x)
    theta = tf.keras.layers.Dense(1, name="rot_factor")(x)
    delta = tf.keras.layers.Dense(2, name="tsl_offset")(x)
    alpha = tf.keras.layers.Dense(1, name="scl_factor")(x)
    output = dict(
        hsv_offset=hsvds, rot_factor=theta, tsl_offset=delta, scl_factor=alpha
    )
    return tf.keras.Model(encoder.input, output, name=encoder.name + "_uspt_xform")


def build_predictor(project_dim, latent_dim, weight_decay=0.5, dropout=0.0):
    layers = [
        tf.keras.layers.Input(shape=[project_dim]),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(
            latent_dim,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        ),
        tf.keras.layers.Activation(tf.nn.silu),
        tf.keras.layers.Dense(project_dim),
    ]
    return tf.keras.Sequential(layers, name="predictor")


class SimSiam(tf.keras.Model):
    def __init__(
        self,
        projector=add_projection_head(build_encoder(), project_dim=512),
        predictor=build_predictor(project_dim=512, latent_dim=256),
        xformer=None,
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
        self.loss_tr = tf.keras.metrics.Mean(name="loss")
        self.build(projector.input.shape)

    @staticmethod
    def cos_dissimilarity(p, z):
        """
        Compute stop-gradient Simple Siamese objective between two latent
        representations.
        """
        p = tf.math.l2_normalize(p, axis=-1)
        z = tf.math.l2_normalize(z, axis=-1)
        return 1 - tf.reduce_mean(tf.reduce_sum(p * z, axis=-1), axis=-1)

    @property
    def metrics(self):
        return [self.loss_tr]

    @property
    def encoder(self):
        return tf.keras.Model(
            self.projector.input,
            self.projector.get_layer("encoding").output,
            name="uspt_encoder",
        )

    def augmented_pairs(self, x):
        if self.xformer is not None:
            u, v = self.xformer(x), self.xformer(x)
            return u, v
        else:
            return x, x

    def train_step(self, data):
        with tf.GradientTape() as tape:
            p = self.augmented_pairs(data)
            z = self.predictor(p[0]), self.predictor(p[1])
            loss = 0.5 * self.cos_dissimilarity(
                p[0], tf.stop_gradient(z[1])
            ) + 0.5 * self.cos_dissimilarity(p[1], tf.stop_gradient(z[0]))
        train = self.projector.trainable_variables + self.predictor.trainable_variables
        grads = tape.gradient(loss, train)
        self.optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, train) if g is not None]
        )
        self.loss_tr.update_state(loss)
        return dict(loss=self.loss_tr.result())

    def call(self, image, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        return self.projector(image, training=training)


class MoCoV2(SimSiam):
    def __init__(self, momentum=1 - 1e-3, temperature=7e-2, max_keys=1024, **kwargs):
        super().__init__(**kwargs)
        projector = self.projector
        kdict_init = tf.random.normal([max_keys, self.projector.output.shape[-1]])
        self.kdict = tf.Variable(initial_value=kdict_init)
        self.projector_q = projector
        self.projector_k = tf.keras.models.clone_model(projector)
        self.projector_k.set_weights(projector.get_weights())
        self.rho = momentum
        self.tau = temperature

    def update_key_encoder(self):
        rho = self.rho
        w_q = self.projector_q.trainable_variables
        w_k = self.projector_k.trainable_variables
        for u, v in zip(w_k, w_q):
            u.assign(u * rho + v * (1 - rho))

    def update_key_dictionary(self, keys):
        # truncate old keys
        self.kdict.assign(self.kdict[tf.shape(keys)[0] :, ...])
        # append new keys
        self.kdict.assign(
            tf.math.l2_normalize(tf.concat([self.kdict, keys], axis=0), axis=-1)
        )

    def contrastive_loss(self, u, v):
        q = tf.math.l2_normalize(self.projector_q(u), axis=-1)
        k = tf.math.l2_normalize(self.projector_k(v), axis=-1)
        batch_size = tf.shape(q)[0]
        pos_logits = q @ tf.transpose(k)
        neg_logits = q @ tf.transpose(self.kdict)
        logits = tf.concat([pos_logits, neg_logits], axis=-1)
        logits = logits * (1 / self.tau)
        labels = tf.zeros(batch_size, dtype=tf.int64)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits) / tf.cast(
            batch_size, dtype=tf.float32
        )
        return loss, q, k

    def symmetric_contrastive_loss(self, u, v):
        l_uv, q_uv, k_uv = self.contrastive_loss(u, v)
        l_vu, q_vu, k_vu = self.contrastive_loss(v, u)
        loss = tf.reduce_mean(l_uv + l_vu)
        qrys = tf.concat([q_uv, q_vu], axis=0)
        keys = tf.concat([k_uv, k_vu], axis=0)
        return loss, qrys, keys

    def train_step(self, data):
        # https://github.com/facebookresearch/moco/blob/main/moco/builder.py
        u, v = self.augmented_pairs(data)
        with tf.GradientTape() as tape:
            loss, qrys, keys = self.symmetric_contrastive_loss(u, v)
            tf.stop_gradient(keys)
        self.update_key_dictionary(keys)
        train = self.projector_q.trainable_variables
        grads = tape.gradient(loss, train)
        self.optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, train) if g is not None]
        )
        self.update_key_encoder()
        self.loss_tr.update_state(loss)
        return dict(loss=self.loss_tr.result())
