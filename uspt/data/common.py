import tensorflow as tf


def coerce_rgb(images):
    return tf.cond(
        tf.shape(images)[-1] < 3,
        lambda: tf.image.grayscale_to_rgb(images[..., :1]),
        lambda: images[..., :3],
    )
