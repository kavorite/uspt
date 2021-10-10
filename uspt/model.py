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


def build_encoder(
    backbone=tf.keras.applications.Xception(
        input_shape=(224, 224, 3), weights=None, include_top=False
    ),
    augmenter=tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomContrast(0.05),
            tf.keras.layers.RandomRotation(0.05),
        ],
        name="data_augmentation",
    ),
):
    inputs = tf.keras.layers.Input(backbone.input.shape[1:])
    outputs = inputs
    if augmenter is not None:
        outputs = augmenter(outputs)
    outputs = backbone(outputs)
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


def build_predictor(project_dim, latent_dim, weight_decay=0.0, dropout=0.0):
    layers = [
        tf.keras.layers.Input(shape=[project_dim]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout),
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
        projector=add_projection_head(build_encoder(), project_dim=2048),
        predictor=build_predictor(project_dim=2048, latent_dim=512),
    ):
        """
        Constructs a SimSiam (simple siamese) unsupervised training harness.

        Parameters
        -------
        projector: Image feature extraction backbone with projection head
        attached.
        predictor: Autoencoder for projected representations.

        References
        -------
        https://arxiv.org/abs/2011.10566
        """
        super().__init__(self)
        self.projector = projector
        self.encoder = tf.keras.Model(
            projector.input,
            projector.get_layer("encoding").output,
            name="uspt_encoder",
        )
        self.predictor = predictor
        self.loss_tr = tf.keras.metrics.Mean(name="loss")
        self.build(projector.input.shape)

    @property
    def metrics(self):
        return [self.loss_tr]

    def train_step(self, data):
        s, t = data
        x = tf.concat([s, t], axis=0)
        with tf.GradientTape() as tape:
            p = tf.math.l2_normalize(self.projector(x), axis=-1, epsilon=1e-9)
            z = tf.math.l2_normalize(self.predictor(p), axis=-1, epsilon=1e-9)
            q, r = tf.split(p, 2, axis=0)
            u, v = tf.split(z, 2, axis=0)
            coses = 0.5 * (q @ tf.transpose(v) + r @ tf.transpose(u))
            error = 1 - tf.reduce_mean(coses)
        train = self.projector.trainable_variables + self.predictor.trainable_variables
        grads = tape.gradient(error, train)
        self.optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, train) if g is not None]
        )
        self.loss_tr.update_state(error)
        return dict(loss=self.loss_tr.result())

    def call(self, image, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        return self.projector(image, training=training)


class DINO(tf.keras.Model):
    def __init__(
        self,
        student_temp=0.1,
        teacher_temp=0.04,
        weight_momentum=1 - 1e-4,
        center_momentum=0.9,
        projector=add_projection_head(build_encoder(), project_dim=2048),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tau_s = student_temp
        self.tau_t = teacher_temp
        self.rho_t = weight_momentum
        self.rho_c = center_momentum
        self.projector = projector
        self.encoder = tf.keras.Model(
            projector.input,
            projector.get_layer("encoding").output,
            name="uspt_encoder",
        )
        self.student = projector
        self.teacher = tf.keras.models.clone_model(projector)
        self.center = tf.Variable(
            initial_value=tf.zeros([1, *projector.output.shape[1:]])
        )
        self.loss_tr = tf.keras.metrics.Mean(name="loss")

    def distill_loss(self, t, s):
        t = tf.stop_gradient(t)
        t = tf.math.l2_normalize(tf.math.divide_no_nan(t - self.center, self.tau_t))
        s = tf.math.l2_normalize(tf.math.divide_no_nan(s, self.tau_s))
        return tf.nn.softmax_cross_entropy_with_logits(t, s)

    def update_teacher(self):
        rho = self.rho_t
        w_s = self.student.trainable_variables
        w_t = self.teacher.trainable_variables
        for u, v in zip(w_t, w_s):
            u.assign(u * rho + v * (1 - rho))

    def update_center(self, keys):
        m = self.rho_c
        u = self.center
        v = tf.reduce_mean(keys, axis=0)
        self.center.assign(m * u + (1 - m) * v)

    def train_step(self, data):
        u, v, *_ = data
        with tf.GradientTape() as tape:
            s = [self.student(x, training=False) for x in data]
            t = [self.teacher(x, training=False) for x in (u, v)]
            error = 0
            error_terms = 0
            for i, k in enumerate(t):
                for j, q in enumerate(s):
                    if i != j:
                        error += tf.reduce_sum(self.distill_loss(k, q))
                        error_terms += 1
            error = tf.math.divide_no_nan(error, error_terms)
        train = self.student.trainable_variables
        grads = tape.gradient(error, train)
        self.optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, train) if g is not None]
        )
        self.loss_tr.update_state(error)
        self.update_teacher()
        self.update_center(tf.concat(t, axis=0))
        return dict(loss=self.loss_tr.result())


class MoCoV2(SimSiam):
    def __init__(
        self,
        projector=add_projection_head(build_encoder(), project_dim=2048),
        momentum=1 - 1e-4,
        temperature=7e-2,
        max_keys=1024,
        **kwargs,
    ):
        super().__init__(**kwargs, predictor=None)
        kdict_init = tf.math.l2_normalize(
            tf.random.normal([max_keys, projector.output.shape[-1]]), axis=-1
        )
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
        trunc = tf.shape(keys)[0]
        kdict = self.kdict[trunc:, ...]
        nkeys = tf.math.l2_normalize(keys, axis=-1)
        self.kdict.assign(tf.concat([kdict, nkeys], axis=0))

    def contrastive_loss(self, q, k):
        pos_logits = q @ tf.transpose(tf.stop_gradient(k))
        neg_logits = q @ tf.transpose(self.kdict)
        batch_size = tf.shape(q)[0]
        logits = tf.math.divide_no_nan(
            tf.concat([pos_logits, neg_logits], axis=-1), self.tau
        )
        labels = tf.zeros(batch_size, dtype=tf.int64)
        errors = tf.math.divide_no_nan(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits),
            tf.cast(batch_size, tf.float32),
        )
        return errors, q, k

    def symmetric_contrastive_loss(self, q, k):
        l_uv, q_uv, k_uv = self.contrastive_loss(q, k)
        l_vu, q_vu, k_vu = self.contrastive_loss(k, q)
        loss = tf.reduce_mean(l_uv + l_vu)
        qrys = tf.concat([q_uv, q_vu], axis=0)
        keys = tf.concat([k_uv, k_vu], axis=0)
        return loss, qrys, keys

    def train_step(self, data):
        s, *t = data
        with tf.GradientTape() as tape:
            p = tf.math.l2_normalize(self.projector_q(s, training=True), axis=-1)
            error = 0
            error_terms = 0
            for v in t:
                q = tf.math.l2_normalize(self.projector_k(v, training=False), axis=-1)
                loss, qrys, keys = self.symmetric_contrastive_loss(p, q)
                error += loss
                error_terms += 1
            error /= error_terms
        self.update_key_dictionary(keys)
        train = self.projector_q.trainable_variables
        grads = tape.gradient(error, train)
        self.optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, train) if g is not None]
        )
        self.update_key_encoder()
        self.loss_tr.update_state(error)
        return dict(loss=self.loss_tr.result())
