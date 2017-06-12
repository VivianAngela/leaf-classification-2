import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img, img_to_array, load_img
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge



def load_preE_training_data(standardize=True):

    """
    This function loads the train.csv file which contains pre extracted features, image ID, and leaf species and returns the ID, train data features, and labels
    """
    
    data = pd.read_csv('train.csv')
    imgID = data.pop('id')
    y = data.pop('species')
    y = LabelEncoder().fit(y).transform(y)
    X = StandardScaler().fit(data).transform(data) if standardize else data.values

    return imgID, X, y


def load_preE_test_data(standardize=True):

    """
    This function loads the test.csv file which contains pre extracted features, and image ids and returns the ID, features, and test data features
    """
    test = pd.read_csv('test.csv')
    imgID = test.pop('id')
    test = StandardScaler().fit(test).transform(test) if standardize else test.values
    return imgID, test


def resize_image(img, max_dim=128):

    """
    This function take an image and dimension as input and outputs the scaled image 128*128
    """

    max_ax = max((0, 1), key=lambda i: img.size[i])
    scale = max_dim / float(img.size[max_ax])
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


def load_image_data_as_numpy_arrays(ids, max_dim=128, center=True):

    """
    This function takes an array of image ids and loads them as numpy arrays and scales the images. When center=True the images will be inserted in the center the outpur array.
    """
    X = np.empty((len(ids), max_dim, max_dim, 1))

    for i, filename in enumerate(ids):

        x = resize_image(load_img(os.path.join('images', str(filename) + '.jpg'), grayscale=True), max_dim=max_dim)
        x = img_to_array(x)
        height = x.shape[0]
        width = x.shape[1]

        if center:

            h1 = int((max_dim - height) / 2)
            h2 = h1 + height
            w1 = int((max_dim - width) / 2)
            w2 = w1 + width

        else:

            h1, w1 = 0, 0
            h2, w2 = (height, width)

        X[i, h1:h2, w1:w2, 0:1] = x
    
	return np.around(X / 255.0)


def split_train_data_into_train_val(split=0.7, random_state=None):

    """
    This function loads the pre-extracted feature and image training data and splits them into training and cross-validation.
    """
    # Load the pre-extracted features
    imgID, Xtr_preE, y = load_preE_training_data()
    # Load the image data
    Xtr_img = load_image_data_as_numpy_arrays(imgID)
    # Split them into validation and cross-validation
    Xtr_preE, Xval_preE, Xtr_img, Xval_img, y_tr, y_val = train_test_split(Xtr_preE, Xtr_img, y, train_size=split, random_state=random_state)
    return (Xtr_preE, Xtr_img, y_tr), (Xval_preE, Xval_img, y_val)

def load_preE_and_image_test():
    """
    This function loads the pre extracted features of test images and test images data and returns the test images ids, pre-extracted test data and test images data.
    """

    imgID, Xte_preE = load_preE_test_data()

    Xte_img = load_image_data_as_numpy_arrays(imgID)

    return imgID, Xte_preE, Xte_img



np.random.seed(8)

print('The training data is loading...')

(Xtr_preE, Xtr_img, y_tr), (Xval_preE, Xval_img, y_val) = split_train_data_into_train_val(random_state=5247)

ytr_cat = to_categorical(y_tr)
yval_cat = to_categorical(y_val)

print('The training data is loaded!')



class ImageDataGenerator2(ImageDataGenerator):

	"""
	This function is used to create artificial data. It create augmented images with rotation range of 10 and zoom range of 0.2.
	"""
    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator2(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class NumpyArrayIterator2(NumpyArrayIterator):
    def next(self):
        with self.lock:
            # We changed index_array to self.index_array
            self.index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(self.index_array):
            x = self.X[j]
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[self.index_array]
        return batch_x, batch_y


print('Data augmentation...')

imgen = ImageDataGenerator2(
    rotation_range=10,
    zoom_range=0.2, 
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')
imgen_train = imgen.flow(Xtr_img, ytr_cat)

print('the data augmenter is created...')




def combined_model():
	
	"""
	This function creates the model in the first row of the TABLE I shown in the final report. Keras is a high level library which allow us to rapidly add or remove a layer and change the number of filters and so on.
	"""

    # Defining the input layer
    image = Input(shape=(128, 128, 1), name='image')

    # Passing the image to the first convolutional layer
    x = Convolution2D(8, 3, 3, subsample=(2,2), input_shape=(128, 128, 1), border_mode='same')(image) 	
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))(x)

    # The second convolutional layer
    x = (Convolution2D(32, 3, 3, subsample=(2,2), border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))(x)


    # Flattening the output of second convolutional layer
    x = Flatten()(x)

    # Defining the pre-extracted feature input
    numerical = Input(shape=(192,), name='numerical')

    # Concatenating the output of convnet with pre-extracted feature input
    concatenated = merge([x, numerical], mode='concat')

    # The first fully conected layer
    x = Dense(512, activation='relu')(concatenated)
    x = Dropout(0.5)(x)

    # The output layer
    out = Dense(99, activation='softmax')(x)

    model = Model(input=[image, numerical], output=out)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

print('Creating the model...')
model = combined_model()
print('Model created!')




def combined_generator(imgen, X):

    """
    This function takes the image augmentor generator and the array of the pre extracted features and yields a minibatch
    """
    while True:
        for i in range(X.shape[0]):

            batch_img, batch_y = next(imgen)
            x = X[imgen.index_array]

            yield [batch_img, x], batch_y

# The best Model is saved automatically
best_model_file = "larnedmodel.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)

start_time = time.time()
print('Training model...')
history = model.fit_generator(combined_generator(imgen_train, Xtr_preE),
                              samples_per_epoch=Xtr_preE.shape[0],
                              nb_epoch=100,
                              validation_data=([Xval_img, Xval_preE], yval_cat),
                              nb_val_samples=Xval_preE.shape[0],
                              verbose=0,
                              callbacks=[best_model, TensorBoard(log_dir='./logs',   						histogram_freq=0, write_graph=True, write_images=False)])

print('Loading the best model...')
model = load_model(best_model_file)
print('Best Model loaded!')
print('')
train_dur = start_time - time.time()
minutes = int(train_dur / 60)
seconds = int(train_dur % 60)
print('Training is completed in %s minutes and %s seconds' %(minutes, seconds))



# Preparing the submission file
LABELS = sorted(pd.read_csv(os.path.join('train.csv')).species.unique())

index, test, Xte_img = load_preE_and_image_test()

yPred_proba = model.predict([Xte_img, test])

# Converting the test predictions format to the sample submission format
yPred = pd.DataFrame(yPred_proba,index=index,columns=LABELS)

print('Writing submission...')
fp = open('submit.csv', 'w')
fp.write(yPred.to_csv())
file.close(fp)
print('Finished writing submission')
print('')


print('Training and validation sets accuracy and loss:')
print('')
print(history.history.keys())

print('acc: ',max(history.history['acc']))
print('loss: ',min(history.history['loss']))

print('val_acc: ',max(history.history['val_acc']))
print('val_loss: ',min(history.history['val_loss']))


# Plotting the history of training loss and validation loss
print('Saving the plot of training and validation loss based on the number of iterations:')
print('') 
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('train_validation_loss.png')



## Plotting the training and validation accuracy
print('Saving the plot of training and validation accuracy based on the number of iterations:')
print('') 

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('train_validation_accuracy.png')


print('')
print('Run the TensorBoard by following command and navigate your browser to localhost:6006:')
print('        tensorboard --logdir=logs    ')



