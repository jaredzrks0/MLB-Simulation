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
    
    # Pull in the raw pitches for dataset building
    file_str = "Data/raw_pitches_2016-2018"

    with open(file_str, 'rb') as file:
        raw_pitches = pkl.load(file)

    # Define the different rolling window settings
    rolling_windows = [#[504],
                       #[75],
                       #[75, 504],
                       #[40, 504],
                       [10, 504],
                       [40, 75],
                       [10, 75],
                       [40, 75, 504],
                       [10, 40, 504],
                       [10, 75, 504],
                       [10, 40, 74, 504]]
    
    for window in rolling_windows:
        print(window)

        # Define the settings for the dataset builder run
        dataset = dataset_builder(rolling_windows=window, verbose=False)
        year_suffix = '2016-2018'
        rolling_window_suffix = '_'.join([str(pa) for pa in dataset.rolling_windows])
        dataset_suffix = f'{year_suffix}_rolling_windows_{rolling_window_suffix}'

        # Create the dataset
        df = dataset.build_training_dataset(raw_pitches, suffix=dataset_suffix,
                                    save_cleaned=False, save_coefficients=False,
                                    save_dataset=False, save_training_dataset=True,
                                    local_save=True, online_save=False)


    





