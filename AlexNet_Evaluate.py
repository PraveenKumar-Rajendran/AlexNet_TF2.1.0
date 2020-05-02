# import necessary package
import tensorflow as tf
import numpy as np
import pathlib
import datetime
# Raw Dataset Directory
data_dir = pathlib.Path("./Test_set")
image_count = len(list(data_dir.glob('*/*.jpg')))
# print total no of images for all classes
print(image_count)
# classnames in the dataset specified
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt" ])
# print list of all classes
print(CLASS_NAMES)
# print length of class names
output_class_units = len(CLASS_NAMES)
print(output_class_units)
#preprocess the data
BATCH_SIZE = 1             # Can be of size 2^n, but not restricted to. for the better utilization of memory
IMG_HEIGHT = 227            # input Shape required by the model
IMG_WIDTH = 227             # input Shape required by the model

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), #Resizing the raw dataset
                                                     classes = list(CLASS_NAMES))
#Loading the saved model
new_model = tf.keras.models.load_model("AlexNet_saved_model/")
new_model.summary()
loss, acc = new_model.evaluate(test_data_gen)
print("accuracy:{:.2f}%".format(acc*100))
