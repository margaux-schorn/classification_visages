# This code come from the Tensorflow github, more precisely :
#   https://github.com/tensorflow/models/tree/master/research/slim
# It has been adapted to suit my project of faces classification.

# ==============================================================================

"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import tensorflow as tf

from datasets import visages
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 60, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'model_name', "inception_v3",
    'The architecture to be used')

tf.app.flags.DEFINE_string(
    'path_to_csv', "labels/labels.csv",
    'The path to csv file'
)

tf.app.flags.DEFINE_string(
    'tfrecord_file', 'labels/labels_records.tfrecord',
    'The path to tfrecord file'
)

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_string(
    'chemin_liste_labels',
    'labels/liste_labels.txt',
    'Path to labels list')

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.checkpoint_path is None:
        raise ValueError("Missing parameter checkpoint_path.")

    with tf.Graph().as_default():
        tf_global_step = tf.train.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = visages.get_split(FLAGS.dataset_split_name,
                                    FLAGS.path_to_csv,
                                    FLAGS.tfrecord_file,
                                    FLAGS.chemin_liste_labels)

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset

        #####################################
        # Select the preprocessing function #
        #####################################
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.model_name, is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(images)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels)
        })

        updates_val = list(names_to_updates.values())
        summary_ops = []

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            summary_ops.append(op)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        summary_op = tf.summary.merge(summary_ops)

        tf.logging.info('Evaluating %s' % checkpoint_path)

        slim.evaluation.evaluation_loop(
            master=FLAGS.master,
            checkpoint_dir=FLAGS.checkpoint_path,
            logdir=FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=updates_val,
            summary_op=summary_op,
            eval_interval_secs=10,
            variables_to_restore=variables_to_restore,
            timeout=60
        )


if __name__ == '__main__':
    tf.app.run()
