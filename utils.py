from pathlib import Path

import tensorflow as tf


def summary_writer(cfg):
    logdir = Path(cfg.dirRuns) / cfg.tag
    writer = tf.summary.create_file_writer(str(logdir))
    writer.set_as_default()
    return writer, logdir


@tf.function
def compute_dice(reference, prediction):
    prediction = tf.cast(tf.argmax(prediction, axis=-1), dtype=tf.float32)
    reference = tf.cast(reference, dtype=tf.float32)
    t1 = tf.reduce_sum(reference * prediction, axis=(1, 2, 3))
    t2 = tf.reduce_sum(reference, axis=(1, 2, 3)) + tf.reduce_sum(prediction, axis=(1, 2, 3))
    dice = (2 * t1 + 1e-8) / (t2 + 1e-8)
    return tf.reduce_mean(dice)


def snapshot(model, save_path, step, override=False):
    if not save_path.exists():
        save_path.mkdir(parents=True)
    filepath = save_path / f"weights-{step}.h5"
    if not filepath.exists() or override:
        model.save_weights(str(filepath))
        return filepath
    return None
