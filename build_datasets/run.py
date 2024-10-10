################################ IMPORTS ################################
# Major Packages
import sys
import pickle as pkl
import warnings
import pandas as pd
from cloud_modules import cloud_functions as cf # type: ignore

# Modules in folder
import constants
import build_datasets as build1
import build_datasets_copy as build2

import time


# Warnings
warnings.simplefilter("ignore")

################################ BUILD DATASET(S) ################################

if __name__ == '__main__':
    
    YEARS = [2016, 2017, 2018]

    # # Import file of raw pitches from the given year
    # raw_pitches = pd.DataFrame()
    # for year in YEARS:
    #     df = cf.CloudHelper().download_from_cloud("yearly_pitches_files/pitches_{}.pkl".format(year))
    #     raw_pitches = pd.concat([raw_pitches, df])

    # with open('Data/test_data_2016-18-long', 'wb') as file:
    #     pkl.dump(raw_pitches, file)

    with open("Data/test_data_2016-18", 'rb') as file:
        raw_pitches = pkl.load(file)

    save = False

    # print("Starting Time Test:")
    # start = time.time()
    # print("\nRunning Original Build")
    # build1.build_training_dataset(raw_pitches, suffix='2016-2018', save_cleaned=save, save_coefficients=save, save_dataset=save, save_training_dataset=save)
    # print(f"Original Code Run Time: {round(time.time() - start, 2)} seconds.")

    start=time.time()
    print("\nRunning Updated Code")
    build2.build_training_dataset(raw_pitches, suffix='2016-2018', save_cleaned=save, save_coefficients=save, save_dataset=save, save_training_dataset=save)
    print(f"Updated Code Run Time: {round(time.time() - start, 2)} seconds.")

    





