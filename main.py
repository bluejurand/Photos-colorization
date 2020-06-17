import os
from tensorflow.keras.callbacks import TensorBoard
import gpu_verification
from generators_definition import image_a_b_gen, encoder_input_generators_definition, tl_model_input_generators_definition
from model_creation import load_transfer_learning_model, build_complete_model
from evaluation import test_model, create_save_images

gpu_verification

batch_size_images = 64
dimension_after_fusion = 256
dataset_images_path = 'D:/Datasets/data_large/'
number_of_images = len(os.walk(dataset_images_path + os.walk(dataset_images_path).__next__()[1][0]).__next__()[2])

transfer_learning_model = load_transfer_learning_model()
train_generator, validation_generator = encoder_input_generators_definition(dataset_images_path, batch_size_images)
train_transfer_learning_generator, validation_transfer_learning_generator = tl_model_input_generators_definition(dataset_images_path, batch_size_images)
model = build_complete_model(dimension_after_fusion)

logs_directory = '.\\TensorboardLogs'
callbacks = [TensorBoard(log_dir=logs_directory, histogram_freq=1, profile_batch=100000000)]

model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
history = model.fit(image_a_b_gen(train_generator, train_transfer_learning_generator, transfer_learning_model),
                    validation_data=image_a_b_gen(validation_generator, validation_transfer_learning_generator, transfer_learning_model),
                    epochs=100,
                    steps_per_epoch=500,#round(number_of_images*0.8) // batch_size_images,
                    validation_steps=50,#round(number_of_images*0.2) // batch_size_images,
                    callbacks=callbacks)
model.save('convolution_xception_fusion_places.h5')

test_images_path = 'C:/Users/mjur1/Documents/Projekty_DS/photos_colorization/test_data/'
save_results_path = './results/image'
files = os.listdir(test_images_path)

for idx, file in enumerate(files):
    test_org, luminance_test, cur = test_model(test_images_path, file, transfer_learning_model, model)
    create_save_images(test_org, luminance_test, cur, save_results_path, idx)
