# Import necessary packages
import os
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from helpers import config


def show_image(row, report=False):
  '''
  Shows given image and reports details of it.

  args:
  image_path -- Path of image file to be showed.
  '''

  row = row.split(',')
  (image_path, startX, startY, w, h) = row

  # lLoad the input image (in OpenCV format), resize it such that it
  # fits on our screen, and grab its dimensions
  image = cv2.imread(image_path)
  (image_h, image_w) = image.shape[:2]
  # sScale the predicted bounding box coordinates based on the image dimensions
  startX = int(float(startX) * image_w)
  startY = int(float(startY) * image_h)
  w = int(float(w) * image_w)
  h = int(float(h) * image_h)

  # draw the predicted bounding box on the image
  cv2.rectangle(image, (startX-int(w/2), startY-int(h/2)), (startX+int(w/2), startY+int(h/2)),
                (0, 0, 255), 1)
  cv2.putText(image, 'Baseball', (startX-int(w/2), startY-int(h/2)-5), cv2.FONT_HERSHEY_SIMPLEX,
              0.5, (0, 0, 255), 1, cv2.LINE_AA)

  if report:  # If report is true
    # Print all image details
    print("Type of the image file: ", image(img))
    print("Format of the image file: ", image.format)
    print("Mode of the image file: ", image.mode)
    print("Size of the image file: ", image.size)

  plt.title("Random sample from dataset")
  plt.imshow(image)


def plot_image_slices(num_rows=4, num_columns=4, slice_size=(8, 8)):
  '''
  Plot a montage of 16 (default) image slices.

  args:
  image_path -- Path of image file set to be showed.
  num_rows -- Number of rows.
  num_columns -- Number of columns.
  image_size -- Size of the image.
  '''

  # Load the contents of the CSV metadata file
  print("[INFO] loading dataset...")
  rows = open(config.METADATA_PATH).read().strip().split('\n')
  # Shuffle rows
  rows = shuffle(rows)

  w, h = slice_size
  fig = plt.figure(figsize=(16, 16))

  for i in range(1, num_columns * num_rows + 1):
    # Select random indices
    random_index = random.randint(0, len(rows)-1)

    fig.add_subplot(num_rows, num_columns, i)
    show_image(rows[random_index], report=False)
    plt.savefig(config.PLOT_PATH + '/random_samples.png')


if __name__ == '__main__':
  plot_image_slices()
