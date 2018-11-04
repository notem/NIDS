from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import os
import pandas as pd
import numpy as np
import keras


def _process_results(truths, preds):
    m = {}
    for truth, pred in list(zip(truths, preds)):
        d = m.get(truth, {})
        d[pred] = d.get(pred, 0) + 1
        m[truth] = d
    return m


class Classifier(object):
    """Common interface for all IDS classifiers """
    trained = False

    def train(self, X_train, Y_train):
        pass

    def evaluate(self, X_test, Y_test):
        pass


class RandomForest(Classifier):
    """RandomForest (supervised) classifier for use in misuse-based NIDS classification"""
    num_trees = 50
    model = None
    le = None
    cm = None

    def train(self, X_train, Y_train):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(Y_train)
        Y_train = self.le.transform(Y_train)

        self.model = RandomForestClassifier(n_jobs=2,
                                            n_estimators=self.num_trees,
                                            oob_score=True,
                                            verbose=True)
        self.model.fit(X_train, Y_train)
        self.trained = True
        print("oob_score = ", self.model.oob_score_)

    def evaluate(self, X_test, Y_test):
        assert (self.trained)
        # make predictions
        results = self.le.inverse_transform(self.model.predict(X_test))
        # create confusion matrix
        self.cm = confusion_matrix(Y_test, results)
        # process results into a dict
        processed = _process_results(Y_test, results)
        return processed


class SVM(Classifier):
    """One-Class SVM (unsupervised) classifier for use in anomaly-based NIDS classification"""
    nu = 0.02
    gamma = 'auto'
    kernel = 'rbf'
    max_iter = -1
    model = None
    cm = None

    def train(self, X_train, Y_train):
        # train a one-class model
        self.model = OneClassSVM(nu=self.nu,
                                 gamma=self.gamma,
                                 max_iter=self.max_iter,
                                 kernel=self.kernel,
                                 verbose=True)
        self.model.fit(X_train)
        self.trained = True

    def evaluate(self, X_test, Y_test):
        assert self.trained
        # make predictions
        results = ['Normal' if r > 0 else 'Attack' for r in self.model.predict(X_test)]
        # create confusion matrix
        self.cm = confusion_matrix(Y_test, results)
        # process results into a dict
        return _process_results(Y_test, results)


class AlexNet(Classifier):
    """Classic (CNN) neural network which performs multi-class (supervised) classification"""
    batch_size = 32
    epochs = 100
    le = None
    cm = None
    model_path = "models/alexnet.h5"

    def train(self, X_train, Y_train):
        if not os.path.exists(self.model_path):
            self.le = preprocessing.LabelEncoder()
            Y_train = self.le.fit_transform(Y_train)

            self._build_model(input_width=len(X_train[0]),
                              class_count=len(set(Y_train)))

            # save best model and logs for tensorboard
            checkpointer = keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                           verbose=0,
                                                           save_best_only=True)
            tensorboard = keras.callbacks.TensorBoard(log_dir='logs',
                                                      histogram_freq=0,
                                                      write_graph=True,
                                                      write_images=True)
            self.model.fit(x=np.array([np.array(x)[..., np.newaxis] for x in X_train]),
                           y=np.array([np.array(y)[..., np.newaxis] for y in Y_train]),
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           shuffle=True,
                           verbose=2,
                           callbacks=[checkpointer, tensorboard])
        # reload the model
        del self.model
        self.model = keras.models.load_model(self.model_path)
        self.trained = True

    def evaluate(self, X_test, Y_test):
        assert self.trained
        predictions = self.model.predict_classes(x=np.array([np.array(x)[..., np.newaxis] for x in X_test]),
                                                 batch_size=self.batch_size,
                                                 verbose=2)
        results = self.le.inverse_transform(predictions)
        self.cm = confusion_matrix(Y_test, results)

        # process results into a dict
        processed = _process_results(Y_test, results)
        return processed

    def _build_model(self, input_width, class_count):
        """keras implementation of the SuperVision NN designed by Alex Krizhevsky et. al.
        NOTE: this implementation deviates from the original design in two ways:
            1) the original was architected to operate distributedly across two systems (this implementation is not distributed)
            2) keras Batch Normalization is used in-place of the original Alexnet's local response normalization
        https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf
        http://image-net.org/challenges/LSVRC/2012/supervision.pdf
        """
        self.model = keras.models.Sequential()
        # layer 1 - "filters the 224 x 224 x 3 input image with 96 kernels
        #           of size 11 x 11 x 3 with a stride of 4 pixels"
        self.model.add(keras.layers.Conv1D(filters=96,
                                           kernel_size=11,
                                           strides=4,
                                           input_shape=(input_width,),
                                           activation="relu",
                                           padding="same"))
        #self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.MaxPool1D(pool_size=3,
                                              strides=2))
        # layer 2 - "256 kernels of size 5 x 5 x 48"
        self.model.add(keras.layers.Conv1D(filters=256,
                                           kernel_size=5,
                                           activation="relu",
                                           padding="same"))
        #self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.MaxPool1D(pool_size=3,
                                              strides=2))
        # layer 3 - "384 kernels of size 3 x 3 x 256"
        self.model.add(keras.layers.Conv1D(filters=384,
                                           kernel_size=3,
                                           activation="relu",
                                           padding="same"))
        # layer 4 - "384 kernels of size 3 x 3 x 192"
        #self.model.add(keras.layers.Conv1D(filters=384,
        #                                   kernel_size=3,
        #                                   activation="relu",
        #                                   padding="same"))
        # layer 5 - "256 kernels of size 3 x 3 x 192"
        #self.model.add(keras.layers.Conv1D(filters=256,
        #                                   kernel_size=3,
        #                                   activation="relu",
        #                                   padding="same"))
        self.model.add(keras.layers.MaxPool1D(pool_size=3,
                                              strides=2))
        # flatten before feeding into FC layers
        self.model.add(keras.layers.Flatten())
        # fully connected layers
        # "The fully-connected layers have 4096 neurons each."
        # "We use dropout in the first two fully-connected layers..."
        self.model.add(keras.layers.Dense(units=4096))  # layer 6
        self.model.add(keras.layers.Dropout(0.5))
        #self.model.add(keras.layers.Dense(units=4096))  # layer 7
        #self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(units=class_count))  # layer 8
        # output layer is softmax
        self.model.add(keras.layers.Activation('softmax'))
        # compile the model with crossentropy loss and SGD
        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer=keras.optimizers.SGD(lr=0.02, momentum=0.9, decay=0.0005),
                           metrics=['accuracy'])


class AutoEncoder(Classifier):
    """AutoEncoder (unsupervised) classifier reconstructs inputs and
    identifies anomalies based on a threshold for the input reconstruction error"""
    encoding_dim = 14
    epochs = 100
    batch_size = 32
    threshold = 2.0
    model_path = "models/autoencoder.h5"

    def train(self, X_train, Y_train):
        if not os.path.exists(self.model_path):
            self._build_model(len(X_train[0]))

            # save best model and logs for tensorboard
            checkpointer = keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                           verbose=0,
                                                           save_best_only=True)
            tensorboard = keras.callbacks.TensorBoard(log_dir='logs',
                                                      histogram_freq=0,
                                                      write_graph=True,
                                                      write_images=True)
            self.model.fit(x=np.array([np.array(x)[..., np.newaxis] for x in X_train]),
                           y=np.array([np.array(x)[..., np.newaxis] for x in X_train]),
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           shuffle=True,
                           validation_split=0.1,
                           verbose=2,
                           callbacks=[checkpointer, tensorboard])
        # reload the best model
        del self.model
        self.model = keras.models.load_model(self.model_path)
        self.trained = True

    def evaluate(self, X_test, Y_test):
        assert self.trained
        predictions = self.model.predict(x=np.array([np.array(x)[..., np.newaxis] for x in X_test]),
                                         batch_size=self.batch_size,
                                         verbose=1)
        mse = np.mean(np.power(X_test - predictions, 2), axis=1)
        error_df = pd.DataFrame({'reconstruction_error': mse,
                                 'true_class': Y_test})
        error_df.describe()

        results = ['Normal' if e < self.threshold else 'Attack' for e in mse]
        # create confusion matrix
        self.cm = confusion_matrix(Y_test, results)
        # process results into a dict
        return _process_results(Y_test, results)

    def _build_model(self, input_width):
        """A simple four layer auto-encoder
        ref: https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras"""
        input_layer = keras.layers.Input(shape=(input_width,))
        encoder = keras.layers.Dense(self.encoding_dim,
                                     activation="tanh",
                                     activity_regularizer=keras.regularizers.l1(10e-5))(input_layer)
        encoder = keras.layers.Dense(int(self.encoding_dim / 2),
                                     activation="relu")(encoder)

        decoder = keras.layers.Dense(int(self.encoding_dim / 2),
                                     activation='tanh')(encoder)
        decoder = keras.layers.Dense(input_width,
                                     activation='relu')(decoder)

        self.model = keras.Model(inputs=input_layer, outputs=decoder)
        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss='mean_squared_error',
                           metrics=['accuracy'])
