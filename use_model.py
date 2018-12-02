from keras.models import model_from_json
import cv2
import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")

'''loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

for i in range(1,10):
    img = cv2.imread("./test/"+ str(i)+".jpg")
    img = cv2.resize(img, (50,50))
    print(img.shape)
    img = img.reshape(1, 50, 50, 3)

    print(img.shape)
    #print(np.argmax(loaded_model.predict(img)))
    prediction = loaded_model.predict(img)
    # print(type(prediction))
    # print(prediction.shape)
    label = ""
    print(prediction)
    print(prediction[0][0])
    if prediction[0][0] > 0.5 :
        label = "dog"
    else:
        label = "cat"
    print("for picture %d.jpg, prediction is: %s" % (i, label))
