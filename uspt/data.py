import math
import os
from typing import Iterable

import tensorflow as tf
import tensorflow_addons as tfa


@tf.keras.utils.register_keras_serializable(package="kavorite/uspt", name="ColorJitter")
class ColorJitter(tf.keras.layers.Layer):
    def __init__(
        self,
        hue_factor=0.4,
        saturation_factor=0.2,
        value_factor=0.4,
        contrast_factor=0.1,
        name="color_jitter",
        seed=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.hue_factor = hue_factor
        self.sat_factor = saturation_factor
        self.val_factor = value_factor
        self.ctr_factor = contrast_factor
        self.seed = seed
        if seed is not None:
            self.rng = tf.random.Generator.from_seed(seed)
        else:
            self.rng = tf.random.Generator.from_non_deterministic_state()

    def call(self, img, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        if not training:
            seeds = self.rng.make_seeds(count=4)
            img = tf.image.stateless_random_brightness(
                img, self.val_factor, seeds[:, 0]
            )
            img = tf.image.stateless_random_contrast(
                img, 1.0 - self.ctr_factor, 1.0 + self.ctr_factor, seeds[:, 1]
            )
            img = tf.image.stateless_random_hue(img, self.hue_factor, seeds[:, 2])
            img = tf.image.stateless_random_saturation(
                img, 1.0 - self.sat_factor, 1.0 + self.sat_factor, seeds[:, 3]
            )
        return img

    def get_config(self):
        base = super().get_config()
        conf = dict(
            hue_factor=self.hue_factor,
            saturation_factor=self.sat_factor,
            value_factor=self.val_factor,
            contrast_factor=self.ctr_factor,
            seed=self.seed,
        )
        base.update(conf)
        return base


@tf.keras.utils.register_keras_serializable(package="kavorite/uspt", name="MultiCrop")
class MultiCrop(tf.keras.layers.Layer):
    def __init__(
        self,
        crop_scale=[0.05, 0.40],
        crop_shape=[224, 224, 3],
        crop_count=8,
        name="multi_crop",
        seed=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.crop_scale = crop_scale
        self.crop_shape = crop_shape
        self.crop_count = crop_count
        self.seed = seed
        if seed is not None:
            self.rng = tf.random.Generator.from_seed(seed)
        else:
            self.rng = tf.random.Generator.from_non_deterministic_state()

    def get_config(self):
        base = super().get_config()
        conf = dict(
            crop_scale=self.crop_scale,
            crop_shape=self.crop_shape,
            crop_count=self.crop_count,
            seed=self.seed,
        )
        base.update(conf)
        return base

    def call(self, img):
        if tf.reduce_any(tf.shape(img)[-3:-1] < self.crop_shape[-3:-1]):
            img = tf.image.resize_with_pad(
                img,
                self.crop_shape[0],
                self.crop_shape[1],
                tf.image.ResizeMethod.BICUBIC,
            )
        seeds = self.rng.make_seeds(count=self.crop_count)
        crops = []
        for i in range(self.crop_count):
            scale = tf.random.stateless_uniform(
                shape=(),
                seed=seeds[:, i],
                minval=self.crop_scale[0],
                maxval=self.crop_scale[1],
            )
            shape = tf.math.round(tf.cast(tf.shape(img)[-3:-1], tf.float32) * scale)
            shape = tf.concat(
                [tf.cast(shape, tf.int32), [self.crop_shape[-1]]], axis=-1
            )
            patch = tf.image.stateless_random_crop(img, shape, seeds[:, i])
            patch = (
                tf.image.resize(
                    patch, self.crop_shape[-3:-1], method=tf.image.ResizeMethod.BICUBIC
                ),
            )
            crops.append(patch)
        return crops


# simplified version of
# https://github.com/facebookresearch/dino/blob/main/main_dino.py
@tf.keras.utils.register_keras_serializable(package="kavorite/uspt", name="DINOAugment")
class DINOAugment(tf.keras.layers.Layer):
    def __init__(
        self,
        crop_scale_factor=2.5,
        global_crop_shape=[224, 224, 3],
        local_crop_shape=[84, 84, 3],
        local_crop_count=8,
        seed=None,
        name="dino_augment",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.crop_scale_factor = crop_scale_factor
        self.global_crop_shape = global_crop_shape
        self.local_crop_shape = local_crop_shape
        self.local_crop_count = local_crop_count
        self.jitter = ColorJitter()
        self.seed = seed
        if seed is not None:
            self.rng = tf.random.Generator.from_seed(seed)
        else:
            self.rng = tf.random.Generator.from_non_deterministic_state()

    def get_config(self):
        base = super().get_config()
        conf = dict(
            crop_scale_factor=self.crop_scale_factor,
            global_crop_shape=self.global_crop_shape,
            local_crop_shape=self.local_crop_shape,
            local_crop_count=self.local_crop_count,
            jitter=self.jitter,
        )
        base.update(conf)
        return base

    def base_augment(self, img):
        img = tf.image.stateless_random_flip_left_right(
            img, self.rng.make_seeds()[:, 0]
        )
        img = tf.cond(
            self.rng.uniform(()) <= 0.8,
            lambda: self.jitter(img),
            lambda: img,
        )
        img = tf.cond(
            self.rng.uniform(()) <= 0.2,
            lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img)),
            lambda: img,
        )
        return img

    def solarize(self, img):
        return tf.where(img < 128, img, 255 - img)

    def blur(self, img):
        def gauss_kernel(channels, kernel_size, sigma):
            ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
            xx, yy = tf.meshgrid(ax, ax)
            kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
            kernel = kernel / tf.reduce_sum(kernel)
            kernel = tf.tile(kernel[..., None], [1, 1, channels])
            return kernel

        kernel = gauss_kernel(
            tf.shape(img)[-1],
            kernel_size=3,
            sigma=self.rng.uniform((), minval=0.1, maxval=2.0),
        )[..., None]
        return tf.nn.depthwise_conv2d(
            img[None, ...], kernel, [1, 1, 1, 1], padding="SAME", data_format="NHWC"
        )[0]

    def call(self, img):
        img_scale = self.rng.uniform((), minval=1.0, maxval=self.crop_scale_factor)
        img_shape = tf.cast(self.global_crop_shape[-3:-1], tf.float32) * img_scale
        img = tf.image.resize(
            img,
            tf.cast(tf.round(img_shape), tf.int32),
            method=tf.image.ResizeMethod.BICUBIC,
        )
        seeds = self.rng.make_seeds(count=2 + self.local_crop_count)
        u = tf.image.stateless_random_crop(img, self.global_crop_shape, seeds[:, 0])
        u = self.base_augment(u)
        u = self.blur(u)

        v = tf.image.stateless_random_crop(img, self.global_crop_shape, seeds[:, 1])
        v = self.base_augment(v)
        v = tf.cond(self.rng.uniform(()) > 0.1, lambda: v, lambda: self.blur(v))
        v = tf.cond(self.rng.uniform(()) > 0.2, lambda: v, lambda: self.solarize(v))

        xs = []
        for i in range(self.local_crop_count):
            x = tf.image.stateless_random_crop(
                img, self.local_crop_shape, seeds[:, i + 2]
            )
            x = self.base_augment(x)
            x = tf.image.resize(x, self.global_crop_shape[-3:-1])
            xs.append(x)
        return (u, v, *xs)


def make_xform_annotator(
    hsv_factor=0.02,
    rot_factor=0.05,
    tls_factor=0.10,
    scl_factor=0.10,
    include_xforms=True,
):
    def xform_and_annotate(x):
        img_size = tf.cast(tf.shape(x)[-3:-1], tf.float32)
        hsvds = tf.random.uniform([3]) * 2.0 - 1.0
        # change hue, saturation, and value
        x = tf.image.rgb_to_hsv(x[..., :3])
        x = tf.clip_by_value(tf.math.abs(x + hsvds * hsv_factor), 0.0, 1.0)
        x = tf.image.hsv_to_rgb(x)

        alpha = tf.random.uniform([])
        tf.image.central_crop(x, alpha * scl_factor)
        alpha = [alpha * 2.0 - 1.0]
        # affine transforms
        theta = tf.random.uniform([1]) * 2.0 - 1.0
        delta = tf.random.uniform([2]) * 2.0 - 1.0
        xforms = [
            tfa.image.angles_to_projective_transforms(
                theta * rot_factor * math.pi, img_size[0], img_size[1]
            ),
            tfa.image.translations_to_projective_transforms(
                delta * img_size * tls_factor
            ),
        ]
        xforms = tfa.image.compose_transforms(xforms)
        x = tfa.image.transform(x, xforms, fill_mode="reflect")
        x = tf.image.random_flip_left_right(x)
        if not include_xforms:
            return x
        else:
            # summarize our changes for the synthetic objective
            y = dict(
                hsv_offset=hsvds,
                rot_factor=theta,
                tsl_offset=delta,
                scl_factor=scl_factor,
            )
            return x, y

    return xform_and_annotate


def make_preprocessor(img_shape, roi_splits=1):
    if isinstance(roi_splits, int):
        roi_splits = [roi_splits] * 2

    roi_splits = tf.convert_to_tensor(roi_splits)
    tile_size = img_shape[:-1]

    @tf.function
    def preprocess(record):
        img = tf.image.decode_jpeg(record["image_str"])
        img = tf.cast(img, tf.float32)
        img = img[..., : img_shape[-1]]
        img = tf.cond(
            tf.shape(img)[-1] != img_shape[-1],
            lambda: tf.image.grayscale_to_rgb(img[..., :1]),
            lambda: img,
        )
        img = tf.image.resize(img, tile_size * roi_splits)
        tiles = tf.image.extract_patches(
            img[None, ...],
            sizes=[1, tile_size[0], tile_size[1], 1],
            strides=[1, tile_size[0] // 2, tile_size[1] // 2, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        tiles = tf.reshape(
            img, tf.concat([[tf.reduce_prod(roi_splits)], img_shape], axis=-1)
        )
        grads = tf.reduce_mean(
            tf.image.sobel_edges(tfa.image.gaussian_filter2d(tiles)), -1 - tf.range(4)
        )
        x = tiles[tf.argmax(grads)]
        return x, record

    return preprocess


def record_parser(feature_desc):
    @tf.function
    def parse(record):
        return tf.io.parse_single_example(record, feature_desc)

    return parse


def read_records(shards, deserialize):
    return (
        tf.data.Dataset.from_tensor_slices(shards)
        .shuffle(len(shards))
        .repeat()
        .interleave(
            lambda shard: (
                tf.data.TFRecordDataset(shard).map(
                    deserialize,
                    num_parallel_calls=1,
                )
            ),
            cycle_length=1,
            block_length=1,
            num_parallel_calls=1,
            deterministic=False,
        )
        .prefetch(tf.data.AUTOTUNE)
    )


def xform_dataset(
    image_shape,
    shards,
    deserialize=record_parser(
        feature_desc=dict(image_str=tf.io.FixedLenFeature((), tf.string))
    ),
    roi_splits=1,
    hsv_factor=0.05,
    tls_factor=0.25,
    rot_factor=0.25,
):
    return (
        read_records(shards, deserialize)
        .map(
            make_preprocessor(image_shape, roi_splits),
            num_parallel_calls=os.cpu_count(),
            deterministic=False,
        )
        .map(lambda x, _: x)
        .map(make_xform_annotator(hsv_factor, tls_factor, rot_factor))
        .apply(tf.data.experimental.ignore_errors())
    )


def supervised_dataset(
    image_shape,
    shards,
    deserialize=record_parser(
        feature_desc=dict(
            image_str=tf.io.FixedLenFeature((), tf.string),
            tag_names=tf.io.FixedLenFeature((), tf.string),
        )
    ),
):
    return (
        read_records(shards, deserialize)
        .map(
            make_preprocessor(image_shape, roi_splits=1),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .apply(tf.data.experimental.ignore_errors())
    )


def contrastive_dataset(
    image_shape,
    shards,
    deserialize=record_parser(
        feature_desc=dict(image_str=tf.io.FixedLenFeature((), tf.string))
    ),
    augment=make_xform_annotator(include_xforms=False),
):
    return (
        read_records(shards, deserialize)
        .map(
            make_preprocessor(image_shape, roi_splits=1),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(lambda x, _: tf.stack((x, x)))
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .map(tf.unstack)
        .apply(tf.data.experimental.ignore_errors())
    )


def dino_dataset(
    image_shape,
    shards,
    deserialize=record_parser(
        feature_desc=dict(image_str=tf.io.FixedLenFeature((), tf.string))
    ),
    **kwargs
):
    return (
        read_records(shards, deserialize)
        .map(
            lambda record: tf.cast(
                tf.image.decode_jpeg(record["image_str"]), tf.float32
            ),
        )
        .map(
            DINOAugment(global_crop_shape=image_shape, **kwargs),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .apply(tf.data.experimental.ignore_errors())
    )


def multicrop_dataset(
    image_shape,
    shards,
    deserialize=record_parser(
        feature_desc=dict(image_str=tf.io.FixedLenFeature((), tf.string))
    ),
    **kwargs
):
    multicrop = MultiCrop(crop_shape=image_shape, **kwargs)
    images = read_records(shards, deserialize).map(
        lambda record: tf.cast(tf.image.decode_jpeg(record["image_str"]), tf.float32)
    )
    resized = images.map(
        lambda img: tf.image.resize(img, image_shape[-3:-1]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    cropped = images.map(multicrop, num_parallel_calls=tf.data.AUTOTUNE)
    return (
        tf.data.Dataset.zip((resized, cropped))
        .map(lambda image, crops: [image, *crops])
        .prefetch(tf.data.AUTOTUNE)
        .apply(tf.data.experimental.ignore_errors())
    )
