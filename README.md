![Elvis](https://github.com/bluejurand/Photos-colorization/blob/master/results/elvis.jpg)  
# Photos colorization
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg) 
![Python 3.6](https://img.shields.io/badge/python-3.3-blue.svg) 
![Spyder 4.1.3](https://img.shields.io/badge/spyder-4.1.3-black) 
![Numpy 1.12.1](https://img.shields.io/badge/numpy-1.12.1-yellow.svg) 
![Matplotlib 2.1.2](https://img.shields.io/badge/matplotlib-2.1.2-blue.svg) 
![Keras 2.3.1](https://img.shields.io/badge/keras-2.3.1-red) 
![Tensorflow 2.1.0](https://img.shields.io/badge/tensorflow-2.1.0-orange) 
![Scikit-image 0.16.2](https://img.shields.io/badge/scikit--image-0.16.2-yellowgreen)  
Presented algorithm is able to colorize black-white photographies. Graph above shows model architecture. Code is implemented in keras API with tensorflow backend.  
Resources which helped to establish this code are listed below, but the main one was deep colorization paper [1].  
The training was done on GPU unit.  

## Motivation

To practice deep learning in keras enviroment, transfer learning and image processing.
Test capabilities of modern algorithms in face of demanding task of image colorization.

## Installation

Python is a requirement (Python 3.3 or greater, or Python 2.7). Recommended enviroment is Anaconda distribution to install Python and Spyder (https://www.anaconda.com/download/).

__Installing dependencies__  
To install can be used pip command in command line.  
  
	pip install -r requirements.txt

__Installing python libraries__  
Exemplary commands to install python libraries:
 
	pip install numpy  
	pip install pandas  
	pip install xgboost  
	pip install seaborn 
	
## Code examples

	def image_a_b_gen(generator, transfer_learning_generator, transfer_learning_model):
		for rgb_image, rgb_tl_image in zip(generator, transfer_learning_generator):
			lab_image = rgb2lab(rgb_image[0])
			luminance = lab_image[:, :, :, [0]]
			ab_components = lab_image[:, :, :, 1:] / 128
			tl_model_features = []
			lab_image_tl = rgb2lab(rgb_tl_image[0])
			luminance_tl = lab_image_tl[:, :, :, [0]]

			for i, sample in enumerate(luminance_tl):
				sample = gray2rgb(sample)
				sample = sample.reshape((1, 331, 331, 3))
				embedding = transfer_learning_model.predict(sample)
				tl_model_features.append(embedding)

			tl_model_features = np.array(tl_model_features)
			tl_model_features_shape_2d = backend.int_shape(Lambda(lambda x: x[:, 0, :], dtype='float32')(tl_model_features))
			tl_model_features = tl_model_features.reshape(tl_model_features_shape_2d)
			yield ([tl_model_features, luminance], ab_components) 
<!-- -->
	def build_encoder(encoder_input):
		encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
		encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
		encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
		encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
		encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
		encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
		encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
		encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
		return encoder_output

## Key Concepts
__CIELab__

__Deep Learning__

__CNNs__

__Transfer Learning__

__Xception__  
https://arxiv.org/pdf/1610.02357.pdf

__cuDNN__
  
![Model architecture](https://github.com/bluejurand/Photos-colorization/blob/master/model_architecture_xception.png)

## Resources
[1] Federico Baldassarre, Diego González Morin, Lucas Rodés-Guirao, *Deep Koalarization: Image Colorization using CNNs and Inception-Resnet-v2*,
(https://arxiv.org/abs/1712.03400)  
[2]