import cv2
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'image_path',
    None,
    'The path to testing image')

tf.app.flags.DEFINE_string(
    'graph_path',
    None,
    'The path to exported model')

tf.app.flags.DEFINE_string(
    'labels_path',
    None,
    'The path to the list of labels')

FLAGS = tf.app.flags.FLAGS

# Read in the image_data
images = []

image = cv2.imread(FLAGS.image_path)
image_data = np.array(image, dtype=np.uint8)
image_data = image_data.astype('float32')
image_data = np.multiply(image_data, 1.0 / 255.0)

image_data = np.resize(image_data, (32, 299, 299, 3))

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile(FLAGS.labels_path)]

with tf.Graph().as_default() as graph:
    # Unpersists graph from file
    with tf.gfile.FastGFile(FLAGS.graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    # Feed the image_data as input to the graph and get first prediction
    with tf.Session(graph=graph) as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('InceptionV3/Predictions/Reshape_1:0')
        predictions = sess.run(softmax_tensor,{'input:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))