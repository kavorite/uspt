import math
import os

import tensorflow as tf
import tensorflow_addons as tfa


def make_xform_annotator():
    def xform_and_annotate(x):
        img_size = tf.cast(tf.shape(x)[-3:-1], tf.float32)
        hsvds = tf.random.uniform([3]) * 2.0 - 1.0
        # change hue, saturation, and value
        x = tf.image.rgb_to_hsv(x[..., :3])
        x = tf.clip_by_value(tf.math.abs(x + hsvds * 0.05), 0.0, 1.0)
        x = tf.image.hsv_to_rgb(x)

        # affine transforms
        theta = tf.random.uniform([1]) * 2.0 - 1.0
        delta = tf.random.uniform([2]) * 2.0 - 1.0
        xforms = [
            tfa.image.angles_to_projective_transforms(
                theta * 0.25 * math.pi, img_size[0], img_size[1]
            ),
            tfa.image.translations_to_projective_transforms(delta * img_size * 0.25),
        ]
        xforms = tfa.image.compose_transforms(xforms)
        x = tfa.image.transform(x, xforms, fill_mode="wrap")

        # summarize our changes for the synthetic objective
        y = dict(hsv_offset=hsvds, rot_factor=theta, tsl_offset=delta)
        return x, y

    return xform_and_annotate


def make_preprocessor(img_shape, roi_splits, top_tiles):
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
        tiles = tf.reshape(
            img, tf.concat([[tf.reduce_prod(roi_splits)], img_shape], axis=-1)
        )
        grads = tf.reduce_mean(tf.image.sobel_edges(tiles), -1 - tf.range(4))
        grads = tf.gather(grads, tf.argsort(grads)[::-1][:top_tiles])
        roi = tf.squeeze(tf.random.categorical(tf.expand_dims(grads, 0), 1))
        return tiles[roi]

    return preprocess


def record_parser():
    feature_desc = dict(
        image_str=tf.io.FixedLenFeature((), tf.string),
        tag_names=tf.io.FixedLenFeature((), tf.string),
    )

    @tf.function
    def parse(record):
        return tf.io.parse_single_example(record, feature_desc)

    return parse


def read_records(shards):
    return (
        tf.data.Dataset.from_tensor_slices(shards)
        .shuffle(len(shards))
        .repeat()
        .interleave(
            lambda shard: (
                tf.data.TFRecordDataset(shard).map(
                    record_parser(),
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


def make_dataset(image_shape, shards, roi_splits=4, top_tiles=4):
    return (
        read_records(shards)
        .map(
            make_preprocessor(image_shape, roi_splits, top_tiles),
            num_parallel_calls=os.cpu_count(),
            deterministic=False,
        )
        .map(make_xform_annotator())
        # .apply(tf.data.experimental.ignore_errors())
    )
