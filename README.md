# Image Classification using TensorFlow for Poets

## Introduction
TensorFlow is an open source software library for machine learning in various kinds of perceptual and language understanding tasks. It is an open source software library for numerical computation using data flow graphs. 

This code retrains Inception, a Google Image Recognition AI that uses a convolutional neural network to find patterns in images. We can retrain the final layer of Inception using custom images. 

This is an example of Inductive Transfer, or transfer learning. Transfer Learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. The inception was already trained on a large number of classes of images. But here, we retrain it on three classes or categories, namely Desktops, Laptops and Tablets.

Note: The images used here are properties of their respective owners and are used for illustration purposes only.

## Requirements
=> Python 2.7 (https://www.python.org/downloads/)
=> Docker (https://docs.docker.com/engine/getstarted/step_one/)
=> Tensor Flow (https://www.tensorflow.org/get_started/os_setup)
=> Tensor Flow for Poets Docker Image (gcr.io/tensorflow/tensorflow:latest-devel)

## Retraining Inception (Optional)

This repo has already a retrained version of Inception, but you can tweak with the arguments and get different results. Also, you can replace the images in tf_files/images/ folder and retrain the model. Just replace the desktops, laptops and tablets folders with your own images.

NOTE: The folder names in which an image is kept is treated as it's label, so place the images accordingly.

To retrain Inception on custom images, start the docker image:
	    sudo docker run -it -v <path to this repo>/tf_files:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel
Example:
	    sudo docker run -it -v $HOME/Programs/machineLearning/tf_files:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel
Then, inside docker, type the following command:
		 python /tensorflow/tensorflow/examples/image_retraining/retrain.py \
		 --bottleneck_dir=/tf_files/bottlenecks \
		 --how_many_training_steps 500 \
		 --model_dir=/tf_files/inception \
		 --output_graph=/tf_files/retrained_graph.pb \
		 --output_labels=/tf_files/retrained_labels.txt \
		 --image_dir /tf_files/images

Change the arguments as desired.
If how_many_training_steps is not given, it will default to 4000 steps. This may take a lot of time, depending on your system.

## Recognising the images
You can test the retrained version of Inception on test images by starting the docker image as:
	    sudo docker run -it -v <path to this folder>:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel
Example:
	    sudo docker run -it -v $HOME/Programs/machineLearning:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel

Then, inside the docker image, type:
		python /tf_files/LabelImage.py /tf_files/test\<name of test file.jpg>
Example:
		python /tf_files/LabelImage.py /tf_files/test\ data/desktop-inline21.jpg

The output will be something like this:
	desktops (score = 0.99620)
	laptops (score = 0.00227)
	tablets (score = 0.00153)
	
That is, it will output the labels and the probability that the image is of that particular label. Here it is 99.6% sure that it is a desktop, which is a correct prediction.

## References
Tensor Flow for Poets (https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#4)
Tensor Flow Wiki (https://en.wikipedia.org/wiki/TensorFlow)
Transfer Learning (https://en.wikipedia.org/wiki/Inductive_transfer)
Tensor Flow Image Retraining (https://www.tensorflow.org/versions/master/how_tos/image_retraining/)
Docker (https://www.docker.com/)
DeepDream (Inception) (https://en.wikipedia.org/wiki/DeepDream)
