import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb, gray2rgb
import matplotlib.pyplot as plt

def test_model(test_path, file, transfer_learning_model, model):
    test_org = img_to_array(load_img(test_path+file))
    test = resize(test_org, (224, 224), anti_aliasing=True)
    test_tl = resize(test_org, (331, 331), anti_aliasing=True)
    test *= 1.0/255
    test_tl *= 1.0/255
    lab_test = rgb2lab(test)
    lab_test_tl = rgb2lab(test_tl)
    luminance_test = lab_test[:, :, 0]
    luminance_test_tl = lab_test_tl[:, :, 0]
    luminance_test_4d = luminance_test.reshape((1, 224, 224, 1))
    embedings_test = gray2rgb(luminance_test_tl)
    embedings_test = embedings_test.reshape((1, 331, 331, 3))
    embedings_test = transfer_learning_model.predict(embedings_test)
    ab_test = model.predict([embedings_test, luminance_test_4d])
    ab_test = ab_test*128
    cur = np.zeros((224, 224, 3))
    cur[:, :, 0] = luminance_test
    cur[:, :, 1:] = ab_test
    cur = resize(cur, (512, 512))
    return test_org, luminance_test, cur

def create_save_images(test_org, luminance_test, cur, save_images_path, idx):
    figure, subplot = plt.subplots(1, 3, figsize=(15, 5))
    subplot[0].imshow(resize(test_org, (512, 512)).astype(np.uint8))
    subplot[0].axis('off')
    subplot[0].title.set_text('orginal')
    subplot[1].imshow(resize(gray2rgb(luminance_test), (512, 512)).astype(np.uint8), cmap='gray')
    subplot[1].axis('off')
    subplot[1].title.set_text('grayscale')
    subplot[2].imshow(lab2rgb(cur))
    subplot[2].axis('off')
    subplot[2].title.set_text('result')
    figure.savefig(save_images_path+str(idx)+".jpg")
    plt.close()
