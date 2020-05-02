######################## AlexNet Architecture Implementation ######################################
#--------------------------------------------------------------------------------------------------
# Praveen Kumar Rajendran | linkedin.com/in/praveenkumar-rajendran/ | aravindhanpraveen19@gmail.com
#--------------------------------------------------------------------------------------------------

# import necessary package
import tensorflow as tf
import numpy as np
import pathlib
import datetime

# printout versions
print(f"Tensor Flow Version: {tf.__version__}")
print(f"numpy Version: {np.version.version}")

# Raw Dataset Directory
data_dir = pathlib.Path("./flower_photos")
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

'''

Stacking of layers, Size of kernels, strides, activations are same as that of in AlexNet paper
BatchNormalization Layer is added to speed up the training process, by limiting covariate shift by normalizing the 
activations. Also BN layers enables much higher learning rates, increasing the speed at which networks train.

'''

model = tf.keras.models.Sequential([
    # 1st conv
  tf.keras.layers.Conv2D(96, (11,11),strides=(4,4), activation='relu', input_shape=(227, 227, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2,2)),
    # 2nd conv
  tf.keras.layers.Conv2D(256, (11,11),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
     # 3rd conv
  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 4th conv
  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 5th Conv
  tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
  # To Flatten layer
  tf.keras.layers.Flatten(),
  # To FC layer 1
  tf.keras.layers.Dense(4096, activation='relu'),
    # add dropout 0.5
  #To FC layer 2
  tf.keras.layers.Dense(4096, activation='relu'),
    # add dropout 0.5
  tf.keras.layers.Dense(output_class_units, activation='softmax')
])


# Shape of inputs to NN Model
BATCH_SIZE = 32             # Can be of size 2^n, but not restricted to. for the better utilization of memory
IMG_HEIGHT = 227            # input Shape required by the model
IMG_WIDTH = 227             # input Shape required by the model
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

# Precprocessing the data
'''
Data preprocessing is one of the important part before doing Deep learning.
And TensorFlow Enables you to do it easily. You can also do Data augmentation on the fly,
Which turns out to help the deep learning model in the task of generalizing well for the unseen data. 
ImageDataGenerator Enables you to do Horizantal/Vertical Flips, Crops etc.
In order fully utilize the ImageDataGenerator refer the TensorFlow documentation.
'''
# Rescalingthe pixel values from 0~255 to 0~1 For RGB Channels of the image.

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# training_data for model training
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), #Resizing the raw dataset
                                                     classes = list(CLASS_NAMES))

# Specifying the optimizer, Loss function for optimization & Metrics to be displayed
# sgd ==> Stochastic Gradient Descent: Explanation by AndrewNg : https://www.youtube.com/watch?v=W9iWNJNFzQI

model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'])

# Summarizing The model architecture and printing it out
model.summary()

# callbacks at training
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get("accuracy")==1.00 and logs.get("loss")<0.03):
            print("\nReached 100% accuracy so stopping training")
            self.model.stop_training =True
callbacks = myCallback()

# TensorBoard.dev Visuals
log_dir="logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training the Model
history = model.fit(
      train_data_gen,
      steps_per_epoch=STEPS_PER_EPOCH,
      epochs=50,
      callbacks=[tensorboard_callback,callbacks])

# Saving the model
model.save('AlexNet_saved_model/')

# Created Link https://tensorboard.dev/experiment/xh8yDX2kR2SvPZgIVqIqNg/
