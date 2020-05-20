import numpy as np
from skimage.color import rgb2lab, gray2rgb
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend

train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

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

def encoder_input_generators_definition(images_path, batch_size_images):
    img_height = 224
    img_width = 224

    train_generator = train_datagen.flow_from_directory(
        images_path,
        target_size=(img_height, img_width),
        batch_size=batch_size_images,
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        images_path,
        target_size=(img_height, img_width),
        batch_size=batch_size_images,
        subset='validation')

    return train_generator, validation_generator

def tl_model_input_generators_definition(images_path, batch_size_images):
    img_transfer_learning_height = 331
    img_transfer_learning_width = 331

    train_transfer_learning_generator = train_datagen.flow_from_directory(
        images_path,
        target_size=(img_transfer_learning_height, img_transfer_learning_width),
        batch_size=batch_size_images,
        subset='training')

    validation_transfer_learning_generator = train_datagen.flow_from_directory(
        images_path,
        target_size=(img_transfer_learning_height, img_transfer_learning_width),
        batch_size=batch_size_images,
        subset='validation')

    return train_transfer_learning_generator, validation_transfer_learning_generator
