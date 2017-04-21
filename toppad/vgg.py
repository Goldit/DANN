from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
tf.logging.set_verbosity(tf.logging.INFO)


from tensorflow.python.framework import dtypes
from random import shuffle

dataset_path       = "toppad/"
train_file         = 'train.txt'
test_file          = 'test.txt'
val_file           = 'val.txt'

NUM_CHANNELS       = 3
BATCH_SIZE         = 64
IMAGE_SIZE         = 256
NUM_CLASSES        = 2
IMG_SIZE_CROPPED   = 224
LEARNING_RATE      = 0.001
DATASET_SIZE       = 20000
STEPS_PER_EPOCH    = int(DATASET_SIZE/BATCH_SIZE)
OPTIMIZER          = 'Adam'
MODEL_NAME         = 'norm'
MODEL              = 'vgg16'
COLOR_DISTORT      = True
DROPOUT_RATE       = 0.0


def encode_label(label):
  return int(label)

def encode_size(size):
    w,h = size.split("x")
    return (int(w), int(h))

def read_image_list(file):
  f = open(file, "r")
  filepaths = []
  labels = []
  for line in f:
    filepath, label, size = line.split(" ")
    filepaths.append(dataset_path + filepath)
    labels.append(encode_label(label))
  return filepaths, labels

def read_images(input_queue):
    file_content = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    label = input_queue[1] 
    return image, label

def distort_color(image):
    # Randomly adjust hue, contrast and saturation.
    if COLOR_DISTORT == True:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    image = tf.divide(image, 255.0)
    if training:
        # For training, add the following to the TensorFlow graph.
        
        # Randomly crop the input image.
        image = tf.random_crop(image, size=[IMG_SIZE_CROPPED, IMG_SIZE_CROPPED, NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        # Distor color
        image = distort_color(image)
    else:
        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=IMG_SIZE_CROPPED,
                                                       target_width=IMG_SIZE_CROPPED)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image
    
def pre_process_image_mean(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 3], name='img_mean')
    
    if training:
        # For training, add the following to the TensorFlow graph.
        
        # Randomly crop the input image.
        image = tf.random_crop(image, size=[IMG_SIZE_CROPPED, IMG_SIZE_CROPPED, NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
    else:
        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=IMG_SIZE_CROPPED,
                                                       target_width=IMG_SIZE_CROPPED)
    image = image - mean
    return image
    

train_images, train_labels = read_image_list(dataset_path + train_file)
test_images, test_labels = read_image_list(dataset_path + test_file)
val_images, val_labels = read_image_list(dataset_path + val_file)


def cnn_model_vgg_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["images"], [-1, IMG_SIZE_CROPPED, IMG_SIZE_CROPPED, NUM_CHANNELS], name="input_layer")

  # Convolutional Layer #1_1
  conv1_1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv1_1")
  # Convolutional Layer #1_2
  conv1_2 = tf.layers.conv2d(
      inputs=conv1_1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv1_2")
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2, name="pool1")
 
  # Convolutional Layer #2_1
  conv2_1 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv2_1")

  # Convolutional Layer #2 and Pooling Layer #2
  conv2_2 = tf.layers.conv2d(
      inputs=conv2_1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv2_2")

  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2, name="pool2")

  # Convolutional Layer #3_1
  conv3_1 = tf.layers.conv2d(
      inputs=pool2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv3_1")

  # Convolutional Layer #3_1
  conv3_2 = tf.layers.conv2d(
      inputs=conv3_1,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv3_2")

  # Convolutional Layer #3_1
  conv3_3 = tf.layers.conv2d(
      inputs=conv3_2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv3_3")

  # Pooling Layer #2
  pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2, name="pool3")    

  # Convolutional Layer #4_1
  conv4_1 = tf.layers.conv2d(
      inputs=pool3,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv4_1")

  # Convolutional Layer #4_2
  conv4_2 = tf.layers.conv2d(
      inputs=conv4_1,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv4_2")

  # Convolutional Layer #4_3
  conv4_3 = tf.layers.conv2d(
      inputs=conv4_2,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv4_3")

  # Pooling Layer #2
  pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], strides=2, name="pool4")

  # Convolutional Layer #5_1
  conv5_1 = tf.layers.conv2d(
      inputs=pool4,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv5_1")

  # Convolutional Layer #5_2
  conv5_2 = tf.layers.conv2d(
      inputs=conv5_1,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv5_2")

  # Convolutional Layer #5_3
  conv5_3 = tf.layers.conv2d(
      inputs=conv5_2,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv5_3")

  # Pooling Layer #2
  pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2, name="pool5")
    
  # FC Layers
  flatten = tf.contrib.layers.flatten(pool5)
  fc1   = tf.layers.dense(inputs=flatten, units=4096, activation=tf.nn.relu, name="fc1")
  dropout1 = tf.layers.dropout(inputs=fc1, rate=DROPOUT_RATE, training=mode == learn.ModeKeys.TRAIN, name="dropout1")
  fc2   = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu, name="fc2")
  dropout2 = tf.layers.dropout(inputs=fc2, rate=DROPOUT_RATE, training=mode == learn.ModeKeys.TRAIN, name="dropout2")

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout2, units=NUM_CLASSES, name="softmax")

  loss = None
  train_op = None

  tf.summary.image('images', features["images"])
  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    with tf.name_scope('loss'):
      onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_CLASSES)
      loss = tf.losses.softmax_cross_entropy(
          onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=LEARNING_RATE,
        optimizer=OPTIMIZER)
  with tf.name_scope('predictions'):
    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)
      
      
def batch_input_fn(images, labels, batch_size=32, training=True, num_epochs=1):  
  with tf.name_scope('input_batch') : 
    # create input queues
    input_queue = tf.train.slice_input_producer(
                        [images, labels],
                        num_epochs=num_epochs,
                        shuffle=True)
  
    image, label = read_images(input_queue)
  
    # preprocess image
    image = pre_process_image(tf.to_float(image), training)
  
    # create batch
    batch_dict = tf.train.batch(dict(images=image, labels=label) , batch_size,
                                num_threads=1, capacity=batch_size*2, 
                                enqueue_many=False, shapes=None, dynamic_pad=False, 
                                allow_smaller_final_batch=False, 
                                shared_name=None, name=None)

    batch_labels = batch_dict.pop('labels')
    return batch_dict, batch_labels
    

# Create the Estimator
toppad_classifier = learn.Estimator(model_fn=cnn_model_vgg_fn, model_dir="log/{}_{}_{}_{}".format(MODEL_NAME, MODEL, OPTIMIZER, LEARNING_RATE))

# Configure validation and test hooks
toppad_validator = learn.monitors.ValidationMonitor(
      input_fn=lambda: batch_input_fn(val_images, val_labels, batch_size=BATCH_SIZE, training=False),
      every_n_steps=STEPS_PER_EPOCH,
      metrics={"accuracy_synthetic": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),},
      early_stopping_metric="loss",
      early_stopping_metric_minimize=True,
      early_stopping_rounds=STEPS_PER_EPOCH*10)

toppad_tester = learn.monitors.ValidationMonitor(
      input_fn=lambda: batch_input_fn(test_images, test_labels, batch_size=BATCH_SIZE, training=False),
      every_n_steps=STEPS_PER_EPOCH,
      metrics={"accuracy_real": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),},)

# Train the model
num_epochs  = 10000
toppad_classifier.fit(
  input_fn=lambda: batch_input_fn(train_images, train_labels, batch_size=BATCH_SIZE, training=True, num_epochs=num_epochs),
  monitors=[toppad_validator, toppad_tester])
