import os
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
saved_model = load_model(os.path.join(APP_ROOT, "model/VGG_model.h5"))
status = True


def check(input_img):
    print("Your image is: " + input_img)

    img_path = os.path.join(APP_ROOT, "static/images", input_img)
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    print(img)

    img = np.expand_dims(img, axis=0)

    print(img)
    output = saved_model.predict(img)

    print(output)
    if output[0][0] > 0.5:
        status = True
    else:
        status = False

    print(status)
    return status
