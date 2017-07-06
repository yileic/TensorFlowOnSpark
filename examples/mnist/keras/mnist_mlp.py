'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from tensorflowonspark import TFCluster, TFNode

import os

def main_fun(args, ctx):
  import numpy
  import tensorflow as tf
  import tensorflow.contrib.keras.api.keras as keras
  from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint, TensorBoard
  from tensorflow.contrib.keras.api.keras import backend as K
  from tensorflow.contrib.keras.api.keras.datasets import mnist
  from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout
  from tensorflow.contrib.keras.api.keras.models import Sequential
  from tensorflow.contrib.keras.api.keras.optimizers import RMSprop

  print("args:",args)
  cluster, server = TFNode.start_cluster_server(ctx)
  job_name = ctx.job_name
  task_index = ctx.task_index

  if job_name == "ps":
    server.join()
  else:
    def generate_local_data(x, y, batch_size):
        print("generate_local_data invoked")
        maxlen = len(x)
        while True:
            i = 0
            while i < (maxlen/batch_size):
                images = x[i*batch_size : (i+1)*batch_size]
                labels = y[i*batch_size : (i+1)*batch_size]
                yield (images, labels)
                i += 1

    def generate_rdd_data(tf_feed, batch_size):
        print("generate_rdd_data invoked")
        while True:
            batch = tf_feed.next_batch(batch_size)
            imgs = []
            lbls = []
            for item in batch:
                imgs.append(item[0])
                lbls.append(item[1])
            images = numpy.array(imgs).astype('float32') / 255
            labels = numpy.array(lbls)
            yield (images, labels)

    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster)):

      num_classes = 10

      # the data, shuffled and split between train and test sets
      (x_train, y_train), (x_test, y_test) = mnist.load_data()

      x_train = x_train.reshape(60000, 784).astype('float32') / 255
      x_test = x_test.reshape(10000, 784).astype('float32') / 255
      print(x_train.shape[0], 'train samples')
      print(x_test.shape[0], 'test samples')

      # convert class vectors to binary class matrices
      y_train = keras.utils.to_categorical(y_train, num_classes)
      y_test = keras.utils.to_categorical(y_test, num_classes)

      model = Sequential()
      model.add(Dense(512, activation='relu', input_shape=(784,)))
      model.add(Dropout(0.2))
      model.add(Dense(512, activation='relu'))
      model.add(Dropout(0.2))
      model.add(Dense(10, activation='softmax'))

      model.summary()

      model.compile(loss='categorical_crossentropy',
          optimizer=RMSprop(),
          metrics=['accuracy'])

    with tf.Session(server.target) as sess:
      K.set_session(sess)

      ckpt_callback = ModelCheckpoint(filepath=os.path.join(args.model, "ckpt.{epoch:02d}.hd5"))
      tb_callback = TensorBoard(log_dir=args.logdir, histogram_freq=1, write_graph=True, write_images=True)

      if args.mode == "tf":
          # load entire dataset in memory and train/validate
          history = model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              verbose=2,
              callbacks=[ckpt_callback, tb_callback],
              validation_data=(x_test, y_test))
#          # use a generator even though entire dataset is in memory
#          history = model.fit_generator(
#              generator=generate_local_data(x_train, y_train, args.batch_size),
#              steps_per_epoch=args.steps_per_epoch,
#              epochs=args.epochs,
#              verbose=2,
#              callbacks=[ckpt_callback, tb_callback],
#              validation_data=(x_test, y_test))
      else:
          # read data from generators/spark for training
          tf_feed = TFNode.DataFeed(ctx.mgr)
          history = model.fit_generator(
              generator=generate_rdd_data(tf_feed, args.batch_size),
              steps_per_epoch=args.steps_per_epoch,
              epochs=args.epochs,
              verbose=2,
              callbacks=[ckpt_callback, tb_callback],
              validation_data=(x_test, y_test))

      score = model.evaluate(x_test, y_test, verbose=0)
      print('Test loss:', score[0])
      print('Test accuracy:', score[1])

      if args.mode == "spark":
          tf_feed.terminate()

if __name__ == '__main__':
    import argparse
    import sys

    sc = SparkContext(conf=SparkConf().setAppName("mnist_mlp"))
    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1
    num_ps = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="number of training samples per batch", type=int, default=100)
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
    parser.add_argument("--epochs", help="number of epochs of training data", type=int, default=20)
    parser.add_argument("--images", help="HDFS path to MNIST images in parallelized CSV format")
    parser.add_argument("--labels", help="HDFS path to MNIST labels in parallelized CSV format")
    parser.add_argument("--logdir", help="directory to write Tensorboard event logs", default="logs")
    parser.add_argument("--mode", help="(spark|tf)", default="tf")
    parser.add_argument("--model", help="directory to write model checkpoints", default="model")
    parser.add_argument("--num_ps", help="number of ps nodes", type=int, default=0)
    parser.add_argument("--steps_per_epoch", help="number of steps per epoch", type=int, default=600)

    args = parser.parse_args()
    print("args:",args)

    if args.mode == "tf":
        cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, args.num_ps, False, TFCluster.InputMode.TENSORFLOW)
    else:
        images = sc.textFile(args.images).map(lambda ln: [int(x) for x in ln.split(',')])
        labels = sc.textFile(args.labels).map(lambda ln: [float(x) for x in ln.split(',')])
        dataRDD = images.zip(labels)
        cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, args.num_ps, False, TFCluster.InputMode.SPARK)
        cluster.train(dataRDD, args.epochs + 1)   # ensure that each executor gets enough data

    cluster.shutdown()
