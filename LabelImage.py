#Uses TensorFlow For Poets Docker Image to test retrained Inception on custom data of desktop,laptop and tablet images.

'''Command to start docker (assuming the training files are in tf_files/images)
        sudo docker run -it -v $HOME/Programs/machineLearning/tf_files:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel

    Inside docker, use the following command to retrain inception:
        python /tensorflow/tensorflow/examples/image_retraining/retrain.py --bottleneck_dir=/tf_files/bottlenecks --ho
        w_many_training_steps 500 --model_dir=/tf_files/inception --output_graph=/tf_files/retrained_graph.pb --output_labels=/tf_files/retrained_labels.txt --image_dir /tf_files/images

    The label given to each image is the name of folder in which it was kept. For example, images in folder ./tf_files/images/desktops will have the label "desktop" and so on
    '''

'''
    Running this example: It can be run by starting the docker as:
        sudo docker run -it -v $HOME/Programs/machineLearning:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel

    And typing the following command in the docker:
        python /tf_files/8.LabelImage.py /tf_files/test\ data/desktop-inline21.jpg
    The output will be the labels (desktops,laptops,tablets) and probability that the image is of a desktop,laptop or tablet.
'''

import tensorflow as tf
import sys

image_path = sys.argv[1] #Pass the test file as argument

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file (the retained labels from retraining) and strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("/tf_files/tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/tf_files/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction i.e. the most likely result
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
