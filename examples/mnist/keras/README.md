# MNIST using Keras

Original Source: https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

This is the MNIST Multi Layer Perceptron example from the [Keras examples](https://github.com/fchollet/keras/blob/master/examples), adapted for TensorFlowOnSpark.

#### Launch the Spark Standalone cluster

    export MASTER=spark://$(hostname):7077
    export SPARK_WORKER_INSTANCES=3
    export CORES_PER_WORKER=1
    export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
    export TFoS_HOME=<path to TensorFlowOnSpark>

    ${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G ${MASTER}

#### Create TFoS zip package

    pushd ${TFoS_HOME}/TensorFlowOnSpark; zip -r ../tfspark.zip *; popd

#### Download the MNIST zip files

    cd ${TFoS_HOME}
    # download the MNIST gzip files into local "mnist" directory

#### Run MNIST MLP using InputMode.TENSORFLOW
In this mode, the workers will load the entire MNIST dataset into memory (directly from zip files) while training.

    # remove any old artifacts, and ensure mnist_model dir exists
    rm -rf ${TFoS_HOME}/mnist_model ${TFoS_HOME}/logs
    mkdir ${TFoS_HOME}/mnist_model

    # train and validate
    ${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --py-files ${TFoS_HOME}/tfspark.zip \
    --conf spark.cores.max=${TOTAL_CORES} \
    --conf spark.task.cpus=${CORES_PER_WORKER} \
    --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
    ${TFoS_HOME}/examples/mnist/keras/mnist_mlp.py \
    --cluster_size 3 \
    --num_ps 1 \
    --epochs 20 \
    --model ${TFoS_HOME}/mnist_model \
    --logdir ${TFoS_HOME}/logs \
    --mode tf

#### Run MNIST MLP using InputMode.SPARK
In this mode, Spark will feed the MNIST dataset (as CSV) into the TF workers.

    # Convert the MNIST zip files into CSV
    # rm -r mnist/csv
    ${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    ${TFoS_HOME}/examples/mnist/mnist_data_setup.py \
    --output mnist/csv \
    --format csv
    ls -lR mnist/csv

    # remove any old artifacts, and ensure mnist_model dir exists
    rm -rf ${TFoS_HOME}/mnist_model ${TFoS_HOME}/logs
    mkdir ${TFoS_HOME}/mnist_model

    # train and validate
    ${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --py-files ${TFoS_HOME}/tfspark.zip \
    --conf spark.cores.max=${TOTAL_CORES} \
    --conf spark.task.cpus=${CORES_PER_WORKER} \
    --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
    ${TFoS_HOME}/examples/mnist/keras/mnist_mlp.py \
    --cluster_size 3 \
    --num_ps 1 \
    --images ${TFoS_HOME}/mnist/csv/train/images \
    --labels ${TFoS_HOME}/mnist/csv/train/labels \
    --epochs 40 \
    --steps_per_epoch 300 \
    --model ${TFoS_HOME}/mnist_model \
    --logdir ${TFoS_HOME}/logs \
    --mode spark

Note: since InputMode.SPARK ends up distributing the dataset across the two workers, we need to adjust the number of epochs and steps per epoch
to match the behavior of the InputMode.TENSORFLOW, where each worker reads the entire dataset per epoch.

#### Shutdown the Spark Standalone cluster

    ${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh

