# Import necessary packages
import os


# Define the base path to the input dataset and then use it to derive the path to the images directory
# and  the annotation CSV file
BASE_PATH = os.getcwd()
DATASET_PATH = os.path.sep.join([BASE_PATH, 'dataset'])
METADATA_PATH = os.path.sep.join([BASE_PATH, 'dataset/baseball.csv'])  # This .csv file contains image paths and their bounding box annotations.

# Define the path to the base output directory
BASE_OUTPUT = os.path.sep.join([BASE_PATH, 'results'])

# Define the path to the output serialized model, model training plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'models/detector.h5'])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'figures'])
TEST_IMAGES_PATH = os.path.sep.join([BASE_OUTPUT, 'test/test_images.txt'])  # This .txt file contains image paths to be tested.

# Initialize our initial learning rate, number of epochs to train for,
# batch size, optimizer decay rate and momentum
INIT_LR = 1e-3
NUM_EPOCHS = 25
BATCH_SIZE = 32
DECAY_RATE = 5e-6
MOMENTUM = 0.90
