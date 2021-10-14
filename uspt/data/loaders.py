import math
import os

import tensorflow as tf
import tensorflow_addons as tfa

from .augmentation import DINOAugment, MultiCrop
from .common import coerce_rgb


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
                )
            ),
            cycle_length=1,
            block_length=1,
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
            DINOAugment(image_size=image_shape[:-1], **kwargs),
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
    multicrop = MultiCrop(crop_dimen=image_shape[-3:-1], **kwargs)
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
