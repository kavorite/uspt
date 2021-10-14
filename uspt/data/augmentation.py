import functools

import tensorflow as tf


def learning_phase(method):
    @functools.wraps(method)
    def learning_phase_op(self, data, training=None, **kwargs):
        if training is None:
            training = tf.keras.backend.learning_phase()
        return tf.cond(
            tf.cast(training, tf.bool),
            lambda: method(self, data, **kwargs),
            lambda: data,
        )

    return learning_phase_op


class Pseudorandom(tf.keras.layers.Layer):
    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        if seed is not None:
            self._rng = tf.random.Generator.from_seed(seed)
        else:
            self._rng = tf.random.Generator.from_non_deterministic_state()

    def get_config(self):
        conf = super().get_config()
        conf.update(dict(seed=self.seed))
        return conf


@tf.keras.utils.register_keras_serializable(
    package="kavorite/uspt", name="WithProbability"
)
class WithProbability(Pseudorandom):
    def __init__(self, inner, p=0.5, name="with_probability", **kwargs):
        super().__init__(name=name, **kwargs)
        self.incidence = p
        self.augment = inner

    def get_config(self):
        conf = super().get_config()
        conf.update(dict(incidence=self.incidence, augment=self.augment))
        return conf

    def call(self, data):
        return tf.cond(
            self._rng.uniform(()) <= self.incidence,
            lambda: self.augment(data),
            lambda: data,
        )


@tf.keras.utils.register_keras_serializable(package="kavorite/uspt", name="ColorJitter")
class ColorJitter(Pseudorandom):
    def __init__(
        self,
        hue_factor=0.4,
        saturation_factor=0.2,
        value_factor=0.4,
        contrast_factor=0.1,
        name="color_jitter",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.hue_factor = hue_factor
        self.sat_factor = saturation_factor
        self.val_factor = value_factor
        self.ctr_factor = contrast_factor

    @learning_phase
    def call(self, image):
        seeds = self._rng.make_seeds(count=4)
        image = tf.image.stateless_random_brightness(
            image, self.val_factor, seeds[:, 0]
        )
        image = tf.image.stateless_random_contrast(
            image, 1.0 - self.ctr_factor, 1.0 + self.ctr_factor, seeds[:, 1]
        )
        image = tf.image.stateless_random_hue(image, self.hue_factor, seeds[:, 2])
        image = tf.image.stateless_random_saturation(
            image, 1.0 - self.sat_factor, 1.0 + self.sat_factor, seeds[:, 3]
        )
        return image

    def get_config(self):
        base = super().get_config()
        conf = dict(
            hue_factor=self.hue_factor,
            saturation_factor=self.sat_factor,
            value_factor=self.val_factor,
            contrast_factor=self.ctr_factor,
        )
        base.update(conf)
        return base


@tf.keras.utils.register_keras_serializable(package="kavorite/uspt", name="RandomBlur")
class RandomBlur(Pseudorandom):
    def __init__(self, min_sigma=0.10, max_sigma=2.00, name="RandomBlur", **kwargs):
        super().__init__(name=name, **kwargs)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def get_config(self):
        base = super().get_config()
        conf = dict(
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
        )
        base.update(conf)
        return base

    def gaussian_kernel(self, depth, width):
        sigma = self._rng.uniform((), minval=self.min_sigma, maxval=self.max_sigma)
        ax = tf.range(-width // 2 + 1.0, width // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., None], [1, 1, depth])
        return kernel[..., None]

    @learning_phase
    def call(self, images):
        kernel = self.gaussian_kernel(depth=tf.shape(images)[-1], width=3)
        images = tf.cond(
            tf.rank(images) < 4,
            lambda: images[None, ...],
            lambda: images,
        )
        return tf.nn.depthwise_conv2d(
            images, kernel, [1, 1, 1, 1], padding="SAME", data_format="NHWC"
        )


@tf.keras.utils.register_keras_serializable(package="kavorite/uspt", name="Grayscale")
class Grayscale(tf.keras.layers.Layer):
    def __init__(self, name="grayscale", **kwargs):
        super().__init__(name=name, **kwargs)

    def __call__(self, images):
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(images))


@tf.keras.utils.register_keras_serializable(
    package="kavorite/uspt", name="RandomSolarize"
)
class RandomSolarize(Pseudorandom):
    def __init__(
        self, min_threshold=128, max_threshold=128, name="RandomSolarize", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def get_config(self):
        conf = super().get_config()
        conf.update(
            dict(min_threshold=self.min_threshold, max_threshold=self.max_threshold)
        )
        return conf

    @learning_phase
    def call(self, images):
        threshold = self._rng.uniform((), self.min_threshold, self.max_threshold)
        return tf.where(images < threshold, images, 255 - threshold)


@tf.keras.utils.register_keras_serializable(package="kavorite/uspt", name="MultiCrop")
class MultiCrop(Pseudorandom):
    def __init__(
        self,
        crop_scale=[0.05, 0.40],
        crop_dimen=[224, 224],
        crop_count=8,
        name="multi_crop",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.crop_scale = crop_scale
        self.crop_dimen = crop_dimen
        self.crop_count = crop_count

    def get_config(self):
        base = super().get_config()
        conf = dict(
            crop_scale=self.crop_scale,
            crop_dimen=self.crop_dimen,
            crop_count=self.crop_count,
        )
        base.update(conf)
        return base

    def call(self, images):
        if tf.reduce_any(tf.shape(images)[-3:-1] < self.crop_dimen):
            images = tf.image.resize_with_pad(
                images,
                self.crop_dimen[0],
                self.crop_dimen[1],
                tf.image.ResizeMethod.BICUBIC,
            )
        seeds = self._rng.make_seeds(count=self.crop_count)
        crops = []
        for i in range(self.crop_count):
            scale = tf.random.stateless_uniform(
                shape=(),
                seed=seeds[:, i],
                minval=self.crop_scale[0],
                maxval=self.crop_scale[1],
            )
            shape = tf.math.round(tf.cast(tf.shape(images)[-3:-1], tf.float32) * scale)
            shape = tf.concat(
                [tf.cast(shape, tf.int32), [tf.shape(images)[-1]]], axis=-1
            )
            shape = tf.concat([tf.shape(images)[:-3], shape], axis=-1)
            patch = tf.image.stateless_random_crop(images, shape, seeds[:, i])
            patch = tf.image.resize(
                patch, self.crop_dimen, method=tf.image.ResizeMethod.BICUBIC
            )
            crops.append(patch)
        return crops


# simplified version of
# https://github.com/facebookresearch/dino/blob/main/main_dino.py
@tf.keras.utils.register_keras_serializable(package="kavorite/uspt", name="DINOAugment")
class DINOAugment(Pseudorandom):
    def __init__(
        self,
        image_size=[224, 224],
        global_crop_scale=[0.40, 1.0],
        local_crop_scale=[0.05, 0.40],
        local_crop_count=8,
        name="dino_augment",
        seed=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.image_size = image_size
        self.local_crop = MultiCrop(
            crop_scale=local_crop_scale,
            seed=seed,
            crop_dimen=image_size,
            crop_count=local_crop_count,
        )
        self.global_crop = MultiCrop(
            crop_scale=global_crop_scale, seed=seed, crop_dimen=image_size, crop_count=2
        )
        self.global_crop_scale = global_crop_scale
        self.local_crop_scale = local_crop_scale
        self.local_crop_count = local_crop_count
        augmentations = [
            tf.keras.layers.RandomFlip(seed=seed, mode="horizontal"),
            WithProbability(
                ColorJitter(seed=seed), p=0.8, seed=seed, name="color_jitter"
            ),
            WithProbability(Grayscale(), p=0.2, name="random_grayscale"),
        ]
        self.flip_color_jitter = tf.keras.Sequential(
            augmentations, name="flip_color_jitter"
        )

        augmentations = [self.flip_color_jitter, RandomBlur(seed=seed)]
        self.global_aug_1 = tf.keras.Sequential(augmentations, name="global_aug_1")

        augmentations = [
            self.flip_color_jitter,
            WithProbability(
                RandomBlur(seed=seed), p=0.1, seed=seed, name="random_blur"
            ),
            WithProbability(RandomSolarize(seed=seed), p=0.2, name="random_solarize"),
        ]
        self.global_aug_2 = tf.keras.Sequential(augmentations, name="global_aug_2")

        augmentations = [
            self.flip_color_jitter,
            WithProbability(
                RandomBlur(seed=seed), p=0.5, seed=seed, name="random_blur"
            ),
        ]
        self.local_aug = tf.keras.Sequential(augmentations, name="local_aug")

    def get_config(self):
        base = super().get_config()
        conf = dict(
            image_size=self.image_size,
            global_crop_scale=self.global_crop_scale,
            local_crop_scale=self.local_crop_scale,
            local_crop_count=self.local_crop_count,
        )
        base.update(conf)
        return base

    def call(self, images):
        images = tf.image.resize(
            images,
            tf.cast(tf.round(self.image_size), tf.int32),
            method=tf.image.ResizeMethod.BICUBIC,
        )
        u, v = self.global_crop(images)
        u = self.global_aug_1(u)
        v = self.global_aug_2(v)
        xs = (self.local_aug(x) for x in self.local_crop(images))
        return (u, v, *xs)
