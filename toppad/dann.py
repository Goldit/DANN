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
from flip_gradient import flip_gradient

dataset_path       = "toppad/"
source_file        = 'source.txt'
target_file        = 'target.txt'
test_file          = 'test.txt'
val_file           = 'val.txt'

NUM_CHANNELS       = 3
BATCH_SIZE         = 64
IMAGE_SIZE         = 256
NUM_CLASSES        = 2
NUM_DOMAINES       = 2
IMG_SIZE_CROPPED   = 224
LEARNING_RATE      = 0.01
DATASET_SIZE       = 40000
STEPS_PER_EPOCH    = int(DATASET_SIZE/BATCH_SIZE)
NUM_EPOCHES        = 100
TOTAL_STEPS        = STEPS_PER_EPOCH * NUM_EPOCHES
OPTIMIZER          = 'Adam'
MODEL_NAME         = 'dann'
LAMBDA             = 10.0
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


source_images, source_labels = read_image_list(dataset_path + source_file)
target_images, target_labels = read_image_list(dataset_path + target_file)
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
    all_features =tf.contrib.layers.flatten(pool5)
    with tf.name_scope('extract_lb_feat'):
        source_features =  tf.strided_slice(all_features, [0], [tf.to_int32(all_features.shape[0])], [2])
        classify_feats = tf.cond(tf.cast(mode == learn.ModeKeys.TRAIN, tf.bool), lambda:source_features, lambda:all_features)
    lb_fc1   = tf.layers.dense(inputs=classify_feats, units=4096, activation=tf.nn.relu, name="lb_fc1")
    lb_dropout1 = tf.layers.dropout(inputs=lb_fc1, rate=DROPOUT_RATE, training=mode==learn.ModeKeys.TRAIN, name="lb_dropout1")
    lb_fc2   = tf.layers.dense(inputs=lb_dropout1, units=4096, activation=tf.nn.relu, name="lb_fc2")
    lb_dropout2 = tf.layers.dropout(inputs=lb_fc2, rate=DROPOUT_RATE, training=mode==learn.ModeKeys.TRAIN, name="lb_dropout2")
    label_logits = tf.layers.dense(inputs=lb_dropout2, units=NUM_CLASSES, name="lb_softmax")

    if mode == learn.ModeKeys.TRAIN:
        with tf.name_scope('flip_gradient'):
            gbt = tf.contrib.framework.get_global_step() 
            p   = tf.cond(tf.cast(gbt == None, tf.bool), lambda:tf.constant(0, tf.float32), lambda:tf.cast(gbt,tf.float32))/TOTAL_STEPS
            l   = 2. / (1. + tf.exp(- LAMBDA * p)) - 1
            flipped_features = flip_gradient(all_features, l)
        dm_fc1   = tf.layers.dense(inputs=flipped_features, units=4096, activation=tf.nn.relu, name="dm_fc1")
        dm_dropout1 = tf.layers.dropout(inputs=dm_fc1, rate=DROPOUT_RATE, training=mode==learn.ModeKeys.TRAIN, name="dm_dropout1")
        dm_fc2   = tf.layers.dense(inputs=dm_dropout1, units=4096, activation=tf.nn.relu, name="dm_fc2")
        dm_dropout2 = tf.layers.dropout(inputs=dm_fc2, rate=DROPOUT_RATE, training=mode==learn.ModeKeys.TRAIN, name="dm_dropout2")
        domain_logits = tf.layers.dense(inputs=dm_dropout2, units=NUM_DOMAINES, name="dm_softmax")

    loss = None
    train_op = None
    
    if mode != learn.ModeKeys.INFER:
        with tf.name_scope('label_loss'):
            source_labels = tf.strided_slice(labels, [0], [tf.to_int32(labels.shape[0])], [2])
            classify_labels = tf.cond(tf.cast(mode == learn.ModeKeys.TRAIN, tf.bool), lambda:source_labels, lambda:labels)
            onehot_labels = tf.one_hot(indices=tf.cast(classify_labels, tf.int32), depth=NUM_CLASSES)
            label_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=label_logits)
            loss = label_loss
    
    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        with tf.name_scope('domain_loss'):
            domain_labels = tf.constant([i%2 for i in range(BATCH_SIZE)], tf.int32)
            onehot_domains = tf.one_hot(indices=tf.cast(domain_labels, tf.int32), depth=NUM_DOMAINES)
            domain_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_domains, logits=domain_logits)
        with tf.name_scope('total_loss'):
            loss = label_loss + domain_loss
            tf.summary.scalar("label_losss", label_loss)
            tf.summary.scalar("domain_loss", domain_loss)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=LEARNING_RATE,
            optimizer=OPTIMIZER)

    with tf.name_scope('predictions'):
        # Generate Predictions
        predictions = {
          "classes": tf.argmax(
              input=label_logits, axis=1),
          "probabilities": tf.nn.softmax(
              label_logits, name="softmax_tensor")
        } 

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)
    
def batch_input_fn(images, labels, batch_size=32, training=True, num_epochs=1):  
    with tf.name_scope('input_batch'):
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

def batch_input_train_fn(source_images, source_labels, target_images, target_labels, batch_size=64, training=True, num_epochs=1):  
    with tf.name_scope('input_batch'):
        # create input queues
        input_source_queue = tf.train.slice_input_producer(
                            [source_images, source_labels],
                            num_epochs=num_epochs,
                            shuffle=True,
                            name='synthetic_image')

        input_target_queue = tf.train.slice_input_producer(
                            [target_images, target_labels],
                            num_epochs=num_epochs,
                            shuffle=True,
                            name='real_image')

        source_image, source_label = read_images(input_source_queue)
        target_image, target_label = read_images(input_target_queue)

        # preprocess image
        source_image = pre_process_image(tf.to_float(source_image), training)
        target_image = pre_process_image(tf.to_float(target_image), training)
        # create batch
        batch_dict = tf.train.batch(dict(images=[source_image, target_image], labels=[source_label, target_label]) , batch_size,
                                    num_threads=1, capacity=batch_size*2, 
                                    enqueue_many=True, shapes=None, dynamic_pad=False, 
                                    allow_smaller_final_batch=False, 
                                    shared_name=None, name=None)
    
    batch_labels = batch_dict.pop('labels')
    return batch_dict, batch_labels
    
# Create the Estimator
toppad_classifier = learn.Estimator(model_fn=cnn_model_vgg_fn, model_dir="log/{}_vgg16_{}_{}".format(MODEL_NAME, OPTIMIZER, LEARNING_RATE))


# Configure validation and test hooks
toppad_validator = learn.monitors.ValidationMonitor(
      input_fn=lambda: batch_input_fn(val_images, val_labels, batch_size=BATCH_SIZE, training=False),
      every_n_steps=STEPS_PER_EPOCH,
      metrics={"accuracy_synthetic": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),},)

toppad_tester = learn.monitors.ValidationMonitor(
      input_fn=lambda: batch_input_fn(test_images, test_labels, batch_size=BATCH_SIZE, training=False),
      every_n_steps=STEPS_PER_EPOCH,
      metrics={"accuracy_real": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),},)
      
# Train the model
toppad_classifier.fit(
  input_fn=lambda: batch_input_train_fn(source_images, source_labels, target_images, target_labels, 
  batch_size=BATCH_SIZE, training=True, num_epochs=NUM_EPOCHES),
  monitors=[toppad_validator, toppad_tester])
