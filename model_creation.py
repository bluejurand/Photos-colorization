from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, RepeatVector, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.utils import  plot_model

def load_transfer_learning_model():
    transfer_learning_model = Xception()
    for layer in transfer_learning_model.layers:
        layer.trainable = False
    return transfer_learning_model

def create_tl_model_embeding(tl_model_embedings):
    tl_model_embedings_2d = RepeatVector(28 * 28)(tl_model_embedings)
    tl_model_embedings_3d = Reshape((28, 28, 1000))(tl_model_embedings_2d)
    return tl_model_embedings_3d

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

def make_fusion(dimension_after_fusion, encoder_output, tl_model_embedings_3d):
    fusion_output = backend.concatenate([encoder_output, tl_model_embedings_3d], axis=3)
    fusion_output = Conv2D(dimension_after_fusion, (1, 1), activation='relu', padding='same')(fusion_output)
    return fusion_output

def build_decoder(fusion_output):
    decoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(fusion_output)
    decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    return decoder_output

def build_complete_model(dimension_after_fusion):
    tl_model_embedings = Input(shape=((1000,)), name='embeding')
    encoder_input = Input(shape=(224, 224, 1,), name='luminance')
    tl_model_embedings_3d = create_tl_model_embeding(tl_model_embedings)
    encoder_output = build_encoder(encoder_input)
    fusion_output = make_fusion(dimension_after_fusion, encoder_output, tl_model_embedings_3d)
    decoder_output = build_decoder(fusion_output)
    model = Model(inputs=[tl_model_embedings, encoder_input], outputs=decoder_output)
    plot_model(model, to_file='model_architecture.png', show_shapes=True)
    return model
