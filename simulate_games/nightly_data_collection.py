import pandas as pd
import numpy as np
import pickle as pkl
import warnings
import datetime
from datetime import datetime as dt

from build_datasets.dataset_builder import DatasetBuilder
from mlb_data_collection.daily_collection import collect_daily_stats
from get_lineups import mlb_scrape

def build_raw_pitches_df(year, month, day, years_prior = 3):
    '''given a year, month, day - stitch together all the raw pitches for the current year plus n years prior'''
    pitches_holder = []

    base_year = int(year) - years_prior
    for n in range(base_year, int(year)+1):
        if n != int(year): # Grab each of the completed pitches files from the previous years
            with open(f'../../../../MLB-Data/raw_pitches/pitches_{n}.pkl', 'rb') as fpath:
                pitches_df = pkl.load(fpath)
                pitches_holder.append(pitches_df)
        else: # Grab the pitches from the current year if applicable, which is titled with a different convention given it is updated daily
            date = f'{year}-{month}-{day}'
            try:
                with open(f'../../../../MLB-Data/raw_pitches/pitches_{year}/pitches_{year}_updated_{date}.pkl', 'rb') as fpath:
                    pitches_df = pkl.load(fpath)
            except FileNotFoundError:
                pitches_df = pd.DataFrame()
            pitches_holder.append(pitches_df)

    raw_pitches = pd.concat([df for df in pitches_holder])
    return raw_pitches

# Define the settings for the dataset builder run
def build_daily_stats_dataset(year, month, day, raw_pitches, windows=(20, 45, 75, 504)):
    dataset = DatasetBuilder(rolling_windows=[window for window in windows], verbose=True)
    date_suffix = f'{year}-{month}-{day}'
    rolling_window_suffix = '_'.join([str(pa) for pa in dataset.rolling_windows])
    dataset_suffix = f'{date_suffix}_rolling_windows_{rolling_window_suffix}'
    
    # Create the dataset
    df = dataset.build_training_dataset(
        raw_pitches,
        suffix=dataset_suffix,
        save_cleaned=False,
        save_coefficients=False,
        save_dataset=True,
        local_save=True,
        online_save=False,
    )

    return df

def build_nightly_stats(year, month, day):
    '''Function to run every night the builds a rolled dataframe for at bat prediction and saves it locally'''
    raw_pitches = build_raw_pitches_df(year, month, day, years_prior=1)
    daily_stats_for_simulation = build_daily_stats_dataset(year, month, day, raw_pitches)

    return daily_stats_for_simulation

### RUN PROGRAM ###
if __name__ == "__main__":
    warnings.simplefilter('ignore')
    today = datetime.datetime.today()
    year, month, day = str(today.year), str(today.month), str(today.day)


    ############### STATS + WEATHER COLLECTION ###############
    # Collect all the new pitches and expected weather for the upcoming day (with saving included)
    collect_daily_stats()

    # Build the daily stats dataframe for PA predicitons
    daily_stats_df = build_nightly_stats(year, month, day)

    ############### LINEUPS COLLECTION ###############
    today = dt.today().strftime('%Y-%m-%d')
    lineups = mlb_scrape(date=today)

    # Save the lineups
    with open(f'../../../../MLB-Data/expected_lineups/expected_lineups_{today}', 'wb') as fpath:
        pkl.dump(lineups, fpath)