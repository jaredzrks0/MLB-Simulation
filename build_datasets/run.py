################################ IMPORTS ################################
# Major Packages
import sys
import pickle as pkl
import warnings
import pandas as pd
from gcloud_helper import cloud_functions as cf

# Modules in folder
import constants
from build_datasets import dataset_builder

import time


# Warnings
warnings.simplefilter("ignore")

################################ BUILD DATASET(S) ################################

if __name__ == '__main__':
    
    file_str = "Data/raw_pitches_2016-2018"

    with open(file_str, 'rb') as file:
        raw_pitches = pkl.load(file)

    save = True

    dataset = dataset_builder()

    year_suffix = '2016-2018'
    rolling_window_suffix = '_'.join([str(pa) for pa in dataset.rolling_windows])

    dataset_suffix = f'{year_suffix}_rolling_windows_{rolling_window_suffix}'

    df = dataset.build_training_dataset(raw_pitches, suffix=dataset_suffix,
                                   save_cleaned=save, save_coefficients=save,
                                   save_dataset=save, save_training_dataset=save,
                                   local_save=True, online_save=False)


    





