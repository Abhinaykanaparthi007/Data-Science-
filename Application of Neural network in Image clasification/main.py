import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from IPython.display import display

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
import matplotlib.image as mpimg
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

digit_train_path = 'data/digit/train.csv'
digit_test_path = 'data/digit/test.csv'
fashion_train_path = 'data/fashion/fashion-mnist_train.csv'
fashion_test_path = 'data/fashion/fashion-mnist_test.csv'

batch_size = 256
epoch = 40

digit_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

fashion_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                 'Ankle boot']


def explore_data(data, target_col, expected_values, rows, cols, figure_size=(8, 8), title='Digit Dataset', *args,
                 **kwargs):
    print('Shape of Input Data', data.shape)
    display(data[target_col].value_counts())
    independent_var = data.drop(target_col, axis=1).values.reshape(-1, 28, 28)
    dependent_var = data[target_col]
    plt.figure(figsize=figure_size)
    for row in range(rows * cols):
        plt.subplot(cols, cols, row + 1)
        plt.yticks([])
        plt.xticks([])
        plt.imshow(independent_var[row], cmap='gray')
        plt.xlabel(expected_values[dependent_var[row]])
        plt.axis(False)
    plt.show()


def create_model(number_of_class=10, name='Digit Dataset', *args, **kwargs):
    if 'first' in kwargs:  # advance Model
        dropout_value = 0.5
        filer_count = 32
        model = Sequential()
        # encoding
        model.add(Conv2D(filters=filer_count, kernel_size=(5, 5), padding='Same',
                         activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(filters=filer_count, kernel_size=(5, 5), padding='Same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # decoding
        model.add(Conv2D(filters=filer_count * 2, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(Conv2D(filters=filer_count * 2, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(dropout_value / 2))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(dropout_value))
        model.add(Dense(number_of_class, activation="softmax"))
    else:  # default model
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        model.add(Flatten())
        model.add(Dense(number_of_class, activation='softmax'))
    return model


def evaluate_model(model, history, X, y, expected_values, *args, **kwargs):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax[1].legend(loc='best', shadow=True)
    plt.show()
    Y_pred = model.predict(X)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(y, axis=1)

    plot_final_images(X, Y_true, Y_pred, 5, 5, expected_values)
    matrix = confusion_matrix(Y_true, Y_pred_classes)
    plot_confusion_matrix(matrix, expected_value=expected_values)


def plot_final_images(X, y, pred, rows, cols, expected_values, figure_size=(18, 16)):
    plt.figure(figsize=figure_size)
    for row in range(rows * rows):
        plt.subplot(cols, cols, row + 1)
        plt.yticks([])
        plt.xticks([])
        value = X[row]
        original_label = expected_values[y[row]]
        pred_label = expected_values[np.argmax(pred[row])]
        value = value.reshape(28, 28)
        plt.imshow(value, cmap='gray')
        plt.title(f'Original Label {original_label}\nPredicted Label {pred_label}')
        plt.axis(False)
    plt.show()


def plot_confusion_matrix(confusion_matrix,
                          expected_value=[]):
    plt.figure(figsize=(10, 10))
    # plt.imshow(confusion_matrix, interpolation='nearest', cmap='Greens')
    sns.heatmap(confusion_matrix, annot=True, cmap='Greens',
                color="gray", fmt='.1f', cbar = False)
    plt.xticks(np.arange(len(expected_value)), expected_value, rotation=45)
    plt.yticks(np.arange(len(expected_value)), expected_value)
    plt.tight_layout()
    plt.ylabel('True Value')
    plt.xlabel('Predicted Value')
    plt.show()


def get_callback_and_optimizer(learning_rate=0.01, rho_value=0.9, epsilon_value=1e-08, patience_level=3, v=1,
                               minimum_learning_rate=0.0001, model_filename='', *args, **kwargs):
    rms_opt = RMSprop(learning_rate=learning_rate, rho=rho_value, epsilon=epsilon_value)
    lr_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                     patience=patience_level,
                                     verbose=v,
                                     min_lr=minimum_learning_rate)

    early_stop = EarlyStopping(patience=patience_level, monitor='val_accuracy')
    model_checkout = ModelCheckpoint(monitor='val_accuracy', save_best_only=True, filepath=model_filename)
    return [rms_opt, lr_reduction, early_stop, model_checkout]


def data_augmentation():
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        rotation_range=8,
        zoom_range=0.2,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
    return datagen


def preprocess_data(data, target_col, n=10, *args, **kwargs):
    if target_col in data.columns:
        X = data.drop(target_col, axis=1).values
        X = X / 255
        X = X.reshape(-1, 28, 28, 1)

        y = to_categorical(data[target_col], num_classes=n)
        print_single_image(X)
        return X, y
    else:
        X = data.values
        X = X / 255
        X = X.reshape(-1, 28, 28, 1)
        print_single_image(X)
        return X, None


def print_single_image(X):
    fig = plt.figure(figsize=(4, 4))
    random_value = random.randint(0, len(X))
    plt.imshow(X[random_value].reshape(28, 28), cmap='gray')
    plt.title('Image')
    plt.axis(False)
    plt.show()


def save_model(model, model_filename, *args, **kwargs):
    try:
        model.save(model_filename)
        print(f'Model is saved at {model_filename}')
    except Exception as e:
        print(f'Error occurred while saving Model {e}')


def train_digit(batch_size=batch_size, epoch=epoch, *args, **kwargs):
    print('Reading Train and Test file for digit dataset')
    digit_train = pd.read_csv(digit_train_path)
    explore_data(digit_train, expected_values=digit_names, target_col='label', rows=5, cols=5, figure_size=(10, 10))
    print('Processing digit dataset')
    train, test = train_test_split(digit_train, train_size=0.3)
    X_train, y_train = preprocess_data(train, 'label')
    X_test, y_test = preprocess_data(test, 'label')
    datagen = data_augmentation()
    print('Model Creation for digit dataset')
    model = create_model(first=True)
    rms_opt, lr_reduction, early_stop, model_checkout = get_callback_and_optimizer(model_filename='digit_model')
    model.compile(optimizer=rms_opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Fitting Model for digit dataset')
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size,
                        callbacks=[lr_reduction, early_stop, model_checkout], validation_data=(X_test, y_test))
    print('Evaluating Model for digit dataset')
    evaluate_model(model, history, X_train, y_train, expected_values=digit_names)
    print('Saving Model for digit dataset')
    save_model(model, model_filename='digit_model')


def train_fashion(batch_size=batch_size, epoch=epoch, *args, **kwargs):
    print('Reading Train and Test file for fashion dataset')
    fashion_train = pd.read_csv(fashion_train_path)
    explore_data(fashion_train, expected_values=fashion_names,
                 target_col='label', rows=5, cols=5, figure_size=(10, 10))
    print('Processing digit dataset')
    train, test = train_test_split(fashion_train, train_size=0.3)
    X_train, y_train = preprocess_data(train, 'label')
    X_test, y_test = preprocess_data(test, 'label')
    datagen = data_augmentation()
    datagen.fit(X_train)
    print('Model Creation for Fashion dataset')
    model = create_model(first=True)
    rms_opt, lr_reduction, early_stop, model_checkout = get_callback_and_optimizer(model_filename='fashion_model')
    model.compile(optimizer=rms_opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Fitting Model for Fashion dataset')
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size,
                        callbacks=[lr_reduction, early_stop, model_checkout], validation_data=(X_test, y_test))
    print('Evaluating Model for Fashion dataset')
    evaluate_model(model, history, X_train, y_train, expected_values=fashion_names)
    print('Saving Model for Fashion dataset')
    save_model(model, model_filename='fashion_model')


def train(*args, **kwargs):
    # to train digit dataset
    train_digit(*args, **kwargs)
    # to train fashion dataset
    train_fashion(*args, **kwargs)


if __name__ == '__main__':
    train()
