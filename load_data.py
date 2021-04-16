# Import necessary packages
import os
import pandas as pd
from helpers import config


def load_data():
  """
  """

  meta_data = []  # List to store meta data informations.

  # Iterate through dataset folder
  for folder in os.listdir(os.path.sep.join([config.DATASET_PATH, 'assignment_sample_data'])):
    # If there is a .txt file for each corresponding .bmp image file
    if len(os.listdir(os.path.sep.join([config.DATASET_PATH, 'assignment_sample_data', folder]))) % 2 == 0:
      # Iterate through subfolder
      for file in os.listdir(os.path.sep.join([config.DATASET_PATH, 'assignment_sample_data', folder])):
        # If the file is an image file (.bmp)
        if file.endswith(".bmp"):
          file = file.split('.')[0]
          annotation_data = []  # Create a list to store annotation data
          # Add its path to `image_path` variable
          image_path = os.path.sep.join([config.DATASET_PATH, 'assignment_sample_data', folder, file + '.bmp'])
          #print(os.path.sep.join([DATASET_PATH, 'assignment_sample_data', folder, file + '.bmp']))
          #print(os.path.sep.join([DATASET_PATH, 'assignment_sample_data', folder, file + '.txt']))

          annotation_data.append(image_path)  # First add corresponding image file's path to the list
          with open(os.path.sep.join([config.DATASET_PATH, 'assignment_sample_data', folder, file + '.txt']), 'rt') as f:
            first_line = f.readline()
            splited = first_line.split();
            try:
                annotation_data.append(float(splited[1]))
                annotation_data.append(float(splited[2]))
                annotation_data.append(float(splited[3]))
                annotation_data.append(float(splited[4]))
                meta_data.append(annotation_data)
            except:
                print("file is not in YOLO format!")

  return meta_data


def create_df(list):
  """
  """

  # Create Pandas data frame
  df = pd.DataFrame(list, columns=['image_path', 'startX', 'startY', 'w', 'h'])
  # Save Pandas data frame as .csv file
  df.to_csv(config.METADATA_PATH, index=False)
  # Show first five records in the data frame
  df.head

  return df


if __name__ == '__main__':
  meta_data = load_data()
  df = create_df(meta_data)
