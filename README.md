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

__Running code__  
Everything is executed from file main.py. Go to directory where code is downoladed and run a command: 

	py main.py
	
Additional requirement is Tensorflow GPU support. Process of configuiring it is described [here](https://www.tensorflow.org/install/gpu).

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
  
## Model architecture  
![Model architecture](https://github.com/bluejurand/Photos-colorization/blob/master/model_architecture_xception.png)

## Results
Below are presented in order from left to right:
- original image, which was an input to the algorithm;  
- grayscale image, it is only the luminance component, one of the outcomes to CIELab transformation;  
- resut image, which is the output of the presented model.  

Resulting images implemented on originaly color photographies:
![Image6](https://github.com/bluejurand/Photos-colorization/blob/master/results/image6.jpg)
![Image12](https://github.com/bluejurand/Photos-colorization/blob/master/results/image12.jpg)
![Image13](https://github.com/bluejurand/Photos-colorization/blob/master/results/image13.jpg)
![Image16](https://github.com/bluejurand/Photos-colorization/blob/master/results/image16.jpg) 
Model implemented on black-white photographies:
![Image17](https://github.com/bluejurand/Photos-colorization/blob/master/results/image17.jpg)  
![Image26](https://github.com/bluejurand/Photos-colorization/blob/master/results/image26.jpg)  
Implementation of algorithm to the historical photo:
![Image21](https://github.com/bluejurand/Photos-colorization/blob/master/results/image21.jpg)  
Couple of photographies which shows the drawbacks of the model. Colorized only part of the images,  
leaving a large parts black and white. Mistakes in color sleection for some of the elements. 
Resulting images are very often faded:
![Image1](https://github.com/bluejurand/Photos-colorization/blob/master/results/image1.jpg)  
![Image9](https://github.com/bluejurand/Photos-colorization/blob/master/results/image9.jpg)  
![Image10](https://github.com/bluejurand/Photos-colorization/blob/master/results/image10.jpg)  
## Summary  


## Resources
[1] Federico Baldassarre, Diego González Morin, Lucas Rodés-Guirao, *Deep Koalarization: Image Colorization using CNNs and Inception-Resnet-v2*,
(https://arxiv.org/abs/1712.03400)  
[2] Richard Zhang, Phillip Isola, Alexei A. Efros, *Colorful Image Colorization*,
(https://arxiv.org/abs/1603.08511)  
[3] Gustav Larsson, Michael Maire, Gregory Shakhnarovich, *Learning Representations for Automatic Colorization*,
(https://arxiv.org/abs/1603.06668)  
[4] Satoshi Iizuka, Edgar Simo-Serra, Hiroshi Ishikawa, *Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors 
for Automatic Image Colorization with Simultaneous Classification*,
(https://www.researchgate.net/publication/305218105_Let_there_be_color_joint_end-to-end_learning_of_global_and_local_image_priors_for_automatic_image_colorization_with_simultaneous_classification)  
[5] https://becominghuman.ai/auto-colorization-of-black-and-white-images-using-machine-learning-auto-encoders-technique-a213b47f7339  
[6] https://fairyonice.github.io/Color-gray-scale-images-and-manga-using-deep-learning.html  