import time
import pandas as pd
import polars as pl
import numpy as np
import pickle as pkl
import os
import sys
import json
import re

 
from scipy import stats
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from datetime import timedelta
from IPython.display import clear_output
from pathlib import Path
from datetime import datetime as dt

from multimodal_communication import cloud_functions as cf
from mlb_simulation.build_datasets import constants
from mlb_simulation.build_datasets.utils_polars import (_correct_home_away_swap,
                   _get_wind_direction,
                   _segregate_plays_by_pitbat_combo
)


class DatasetBuilder():

    def __init__(self, rolling_windows=[75, 504], verbose=False,  gcloud_upload=False,
                 gcloud_upload_path='', local_save=False, local_save_dir_path=''):
        
        self.rolling_windows = rolling_windows
        self.verbose = verbose
        self.gcloud_upload = gcloud_upload
        self.gcloud_upload_path = gcloud_upload_path
        self.local_save = local_save
        self.local_save_dir_path = local_save_dir_path

    def build_training_dataset(self, raw_pitches, save_coefficients=False, coef_save_path=''):
        """
        Cleans raw pitch data, generates neutralization coefficients, anad build a final
        machine readable dataset.

        Args:
            raw_pitches (dict): Raw pitch data for each 'pitbat' combo.
            suffix (str): Suffix for file names.
            save_coefficients (bool): Whether to save neutralization coefficients.

        Returns:
            dict: Training dataset dictionary containing features and target values.

        FUNCTION CONNECTIONS:
        ----------------------
        Calls On: _clean_raw_pitches()
                  _build_neutralization_coefficient_dictionaries()
                  _make_final_dataset()
        """

        # Clean raw pitches and return a cleaned pitches DataFrame
        cleaned_data = self._clean_raw_pitches(raw_pitches)

        # Create a neutralization coefficients dictionary
        coef_dicts = self.build_neutralization_coefficient_dictionaries(cleaned_data)

        if save_coefficients:
            # Format the windows in a variable to help with the naming conventions while saving
            windows = '_'.join([window for window in self.rolling_windows])

            if self.gcloud_upload:
                cf.CloudHelper(obj=coef_dicts).upload_to_cloud(
                    'simulation_training_data', f"neutralization_coefficients_dict_{windows}")
            if self.local_save:
                if not coef_save_path: # Ensure a path is given for a local save
                    raise ValueError('In order to save the coefficients locally, a path to a directory must be provided')
                
                base_path = Path(coef_save_path)
                filename = f'/neutralization_coefficients_dict_{windows}.pkl'
                full_path = base_path + filename
                base_path.mkdir(parents=True, exist_ok=True)

                with open(full_path, 'wb') as f:
                    pkl.dump(coef_dicts, f)

        # Build the final dataset
        final_dataset = self._make_final_dataset(cleaned_data, coef_dicts)
        if self.gcloud_upload:
            cf.CloudHelper(obj=final_dataset).upload_to_cloud(
                'simulation_training_data', f"Final Datasets/final_dataset_{windows}")
        if self.local_save:
            base_path = Path(self.local_save_dir_path)
            filename = f'/daily_stats_df_updated_{dt.today().strftime("%Y-%m-%d")}.pkl'
            full_path = base_path + filename
            base_path.mkdir(parents=True, exist_ok=True)

            with open(full_path, 'wb') as f:
                pkl.dump(final_dataset, f)
        
        return final_dataset
    
    ######################################################################################
    # Clean Pitch Data
    ######################################################################################
    def clean_raw_pitches(self, raw_pitches_df: pd.DataFrame) -> pl.LazyFrame:
        """
        Cleans a DataFrame of raw pitch data, filtering and transforming it into a usable format 
        for subsequent analyses, including attaching weather and ballpark information.

        Parameters:
            raw_pitches_df (DataFrame): A DataFrame of uncleaned pitch data from the Statcast API.

        Returns:
            dict: A dictionary with 4 keys ("RR", "RL", "LR", "LL"), each containing a DataFrame 
            of pitches divided by batter-pitcher handedness combination.

        FUNCTION CONNECTIONS:
        ----------------------
        Calls On: 
        """

        if self.verbose:
            print("Cleaning Data")

        # Grab necesssary information from the df for later use that would
        # later require calling collect early
        self.unique_years = raw_pitches_df.game_date.dt.year.unique().tolist()

        # Convert the raw_pitches file to a LazyFrame
        raw_pitches_df = pl.from_pandas(raw_pitches_df).lazy()

        # Filter down to only regular season games
        raw_pitches_df = raw_pitches_df.filter(pl.col('game_type') == 'R')
    
        # Correct home and away mistakes in the pitch data
        #raw_pitches_df = _correct_home_away_swap(raw_pitches_df)

        # Convert the datetime game_date to a string formatted as YYYY-MM-DD, and sort the df on the column to make sure everything is in order
        raw_pitches_df = raw_pitches_df.with_columns(
            pl.col("game_date").dt.strftime("%Y-%m-%d").alias("game_date")
        ).sort(by=["game_date", "inning", "inning_topbot", "at_bat_number"],
               descending=[False, False, False, False])

        # Filter all pitches to only those with an event\
        raw_plays = raw_pitches_df.drop_nulls(subset=['events'])

        # Filter all pitches with an event to only those types we care about
        # As well as only the columns we care about
        final_plays = raw_plays.filter(
            pl.col('events').is_in(constants.RELEVANT_PLAY_TYPES)
        ).select(
            constants.RELEVANT_BATTING_COLUMNS
        )

        # Add a new column that groups all the event types into eventual Y labels
        final_plays = final_plays.with_columns(
            pl.col('events').replace(constants.PLAY_TYPE_DICT).alias('play_type')
        )

        # Insert a new 'type counter' coulumn that will be used repeatedly for calculating rolling stats
        final_plays = final_plays.with_columns(
            pl.lit(1).alias('type_counter')
        )
        

        ############ ATTATCH WEATHER INFORMATION TO EACH PITCH ############
        
        weather_dictionary_holder = {}

        for year in self.unique_years:
            # Pull in the proreference weather data 
            yearly_weather_df = pd.DataFrame()
            base_path = self.local_save_dir_path
            filename = f'/proreference_weather_data/weather_data_{year}.pkl'
            weather_filepath = base_path + filename
            if os.path.exists(weather_filepath):
                with open(weather_filepath, 'rb') as fpath:
                    yearly_weather_df = pl.read_csv(fpath)
            
            # If not locally, download from the cloud
            if len(yearly_weather_df) == 0:
                yearly_weather_df = cf.CloudHelper().download_from_cloud("proreference_weather_data/weather_data_{}".format(year))
                yearly_weather_df = pl.from_pandas(yearly_weather_df)  

            if len(yearly_weather_df) == 0:
                raise Exception(f'No Prorefence weather data was found for {year}. Ensure that the data exists and paths are correct!')
            
            # Insert each years data into storage as a LazyFrame
            weather_dictionary_holder[year] = yearly_weather_df.lazy()
        
        # Concat all the yearly weather dataframes into one larger one
        total_weather_df = pl.concat([df for df in weather_dictionary_holder.values()], how='vertical')

        # Create a new column with the converted home team names in total_weather_df
        team_name_map = {v: k for k, v in constants.WEATHER_NAME_CONVERSIONS.items()}
        total_weather_df = total_weather_df.with_columns([
            pl.col('home_team').replace(team_name_map).alias('converted_home_team'),
            pl.col('away_team').replace(team_name_map).alias('converted_away_team')
        ])

        # Drop the old columns
        total_weather_df = total_weather_df.drop(['home_team', 'away_team'])

        # Attatch the full str description of the weather, found by joining the pitches df and the weather df
        # Note that this will add atrifically add some rows due to double headers, that will join with both
        # Descriptions, but this can be thought of an 'average' of the two
        total_weather_df = total_weather_df.rename({'date':'game_date', 'converted_home_team':'home_team'})
        final_plays = final_plays.join(total_weather_df, on=['game_date', 'home_team']).drop(
            ['converted_away_team', 'url']
        )
        
        # Break up the full weather info into temp, wind speed, and wind direction seperately
        final_plays = final_plays.with_columns(
            pl.col('weather').str.split(': ').list.get(1).str.split("°").list.get(0).cast(pl.Int64).alias('temperature'),
            
            # Wind direction
            pl.when(pl.col("weather").is_not_null())
            .then(
                pl.when(pl.col("weather").str.replace("Wind", "").str.contains("(?i)in"))
                    .then(pl.lit("in"))
                .when(pl.col("weather").str.replace("Wind", "").str.contains("(?i)out"))
                    .then(pl.lit("out"))
                .when(pl.col("weather").str.contains("Left|Right"))
                    .then(pl.col("weather").str.extract(r"from ([^,.]+)", 1).str.to_lowercase())
                .otherwise(None)
            )
            .otherwise(None)
            .alias("wind_direction"),

            # Wind Speed
            pl.when(pl.col('weather').str.contains("Wind"))
            .then(
                pl.col('weather').str.split('Wind ').list.get(1).str.split('mph').list.get(0).cast(pl.Int64) 
            )
            .otherwise(pl.lit(0))
            .alias('wind_speed')
        )

        # Convert the wind direction for 0 wind results from 'in' to 'zero'
        final_plays = final_plays.with_columns(
            pl.when(pl.col('wind_speed') == 0)
                .then(pl.lit('zero'))
                .otherwise(pl.col('wind_direction'))
                .alias('wind_direction')
        )

        # Create columns for each of the wind directions, with the value of the wind speed in that direction
        final_plays = final_plays.with_columns(
            (pl.col('wind_speed') * (pl.col('wind_direction').eq(pl.lit('in')))).alias('in'),
            (pl.col('wind_speed') * (pl.col('wind_direction').eq(pl.lit('out')))).alias('out'),
            (pl.col('wind_speed') * (pl.col('wind_direction').eq(pl.lit('right to left')))).alias('rtl'),
            (pl.col('wind_speed') * (pl.col('wind_direction').eq(pl.lit('left to right')))).alias('ltr')
        ).drop('wind_direction', 'wind_speed')

        
        ############ ATTATCH BALLPARK INFO TO EACH PITCH ############

        # Import file to help connect team and year with a specific ballpark
        ballpark_pandas = pd.read_excel('data/Ballpark Info.xlsx')
        ballpark_info = pl.from_pandas(ballpark_pandas).lazy()

        # Join the pitches to the ballparks
        final_plays = final_plays.join(ballpark_info, on='home_team', how='left')
        final_plays = final_plays.filter(
            pl.col('game_date').str.split('-').list.get(0).cast(pl.Int64) < pl.col('End Date').cast(pl.Int64)
        )

        ############ Divide pitches by pitbat combos in 4 dataframes ############
        all_plays_by_pitbat_combo = {}
        for pitbat_combo in ['RR', 'RL', 'LR', 'LL']:
            all_plays_by_pitbat_combo[pitbat_combo] = final_plays.filter(
                (pl.col('stand').eq(pitbat_combo[0])) &
                (pl.col('p_throws').eq(pitbat_combo[1]))
            )
        

        return all_plays_by_pitbat_combo
