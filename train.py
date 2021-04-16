import numpy as np
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50, DenseNet121, MobileNet, MobileNetV2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, SGD, RMSprop
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from helpers import config
from tqdm import tqdm


def load_data():
    '''
    '''

    # Load the contents of the CSV metadata file
    print("[INFO] loading dataset...")
    rows = open(config.METADATA_PATH).read().strip().split('\n')

    # Initialize the list of data (images), our target output predictions (bounding box coordinates),
    # along with the filenames of the individual images
    data = []
    targets = []
    image_paths = []

    # Loop over the rows
    for row in tqdm((rows[1:])):
      # Break the row into the filename and bounding box coordinates
      row = row.split(',')
      (image_path, startX, startY, w, h) = row

      # Load the image and preprocess it
      image = load_img(image_path, target_size=(224, 224))  #, color_mode='grayscale',
      image = img_to_array(image)
      # Update our list of data, targets, and image_paths
      data.append(image)
      targets.append((startX, startY, w, h))
      image_paths.append(image_path)



    # Convert the data and targets to NumPy arrays, scaling the input
    # pixel intensities from the range [0, 255] to [0, 1]
    data = np.array(data, dtype='float32') / 255.0
    targets = np.array(targets, dtype='float32')
    # partition the data into training and testing splits using 90% of
    # the data for training and the remaining 10% for testing
    split = train_test_split(data, targets, image_paths, test_size=0.10, random_state=42)

    return split


def save_testing_images(testFilenames):
    '''
    '''

    # Write the testing filenames to disk so that we can use then
    # when evaluating/testing our bounding box regressor
    print("[INFO] saving testing image names...")
    print(testFilenames)
    f = open(config.TEST_IMAGES_PATH, 'w')
    f.write('\n'.join(testFilenames))
    f.close()


def build_model():
    '''
    '''

    # Load the MobileNetV2 network, ensuring the head FC layers are left off
    mobilenetv2 = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    # Freeze all MobileNetV2 layers so they will *not* be updated during the training process
    mobilenetv2.trainable = False
    # flatten the max-pooling output of VGG
    flatten = mobilenetv2.output
    flatten = Flatten()(flatten)
    # Construct a fully-connected layer header to output the predicted bounding box coordinates
    bboxHead = Dense(32, activation="relu")(flatten)
    bboxHead = Dropout(0.3)(bboxHead)
    bboxHead = Dense(4, activation="sigmoid")(bboxHead)
    # Construct the model we will fine-tune for bounding box regression
    model = Model(inputs=mobilenetv2.input, outputs=bboxHead)

    return model


def train_model(model, trainImages, trainTargets, testImages, testTargets):
    '''
    '''

    # Initialize the optimizer, compile the model, and show the model
    # Summary
    opt = SGD(lr=config.INIT_LR, decay=config.DECAY_RATE, momentum=config.MOMENTUM)
    model.compile(loss="mse", optimizer=opt)
    #print(model.summary())

    # train the network for bounding box regression
    print("[INFO] training bounding box regressor...")

    history = model.fit(trainImages, trainTargets,
      validation_data=(testImages, testTargets),
      batch_size=config.BATCH_SIZE,
      epochs=config.NUM_EPOCHS,
      verbose=1)

    print("[INFO] saving object detector model...")
    model.save(config.MODEL_PATH, save_format='h5')

    return history


if __name__ == '__main__':
    split = load_data()
    (trainImages, testImages) = split[:2]
    (trainTargets, testTargets) = split[2:4]
    (trainFilenames, testFilenames) = split[4:]
    save_testing_images(testFilenames)
    model = build_model()
    history = train_model(model, trainImages, trainTargets, testImages, testTargets)
