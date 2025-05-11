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
        coef_dicts = self._build_neutralization_coefficient_dictionaries(cleaned_data)

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
    def _clean_raw_pitches(self, raw_pitches_df: pd.DataFrame) -> pl.LazyFrame:
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
        ).drop('Start Date', 'End Date', 'Full Name')

        ############ Divide pitches by pitbat combos in 4 dataframes ############
        all_plays_by_pitbat_combo = {}
        for pitbat_combo in ['RR', 'RL', 'LR', 'LL']:
            all_plays_by_pitbat_combo[pitbat_combo] = final_plays.filter(
                (pl.col('stand').eq(pitbat_combo[0])) &
                (pl.col('p_throws').eq(pitbat_combo[1]))
            )
        
        return all_plays_by_pitbat_combo
    
    #######################################################################################################################
    # Build weather coefficients and ballpark coeficients dictionaries, which will be used in neutralization
    #######################################################################################################################

    def _insert_game_play_shares(self, all_plays_by_pitbat_combo: dict) -> dict:
        """
        Calculates the game play share (percentage of play type outcomes) for each game and inserts 
        the share as a new column into the relevant DataFrames within the input dictionary.

        Parameters:
            all_plays_by_pitbat_combo (dict): A dictionary of DataFrames, each containing pitches 
                divided by batter-pitcher handedness combination, including columns for play type and game identifier.

        Returns:
            dict: A dictionary with the same keys as the input, but each DataFrame now includes a 
            new column `game_play_share`, representing the percentage of each play type in each game.
        """

        weather_regression_dfs = {x: {} for x in constants.HAND_COMBOS}

        for pitbat_combo in constants.HAND_COMBOS:
            full_df = all_plays_by_pitbat_combo[pitbat_combo]

            # Count the number of total plays per game of each type
            plays_per_game_by_type = (
                full_df
                .group_by(['game_pk', 'play_type'])
                .agg([
                    pl.len().alias('play_count')
                ])
            )

            # Count the total number of plays per game
            total_plays_per_game = (
                full_df
                .group_by(['game_pk'])
                .agg([
                    pl.len().alias('total_game_plays')
                ])
            )

            # Join the two tables
            play_shares_by_game = (
                plays_per_game_by_type
                .join(total_plays_per_game, on='game_pk', how='left')
                .with_columns(
                    (pl.col('play_count') / pl.col('total_game_plays')).alias('game_play_share')
                )
                .cache() # Cache the result for use in building out the weather regression df
            )

            unique_plays = play_shares_by_game.select('play_type').unique()
            unique_games = play_shares_by_game.select('game_pk').unique()

            weather_regression_df = (
                unique_games
                .join(unique_plays, how='cross') # Compute all game/play type combos
                .join(play_shares_by_game, how='left', on=['game_pk', 'play_type']) # Join each combo with the play share
                .fill_null(0.0)
                .drop(['play_count', 'total_game_plays'])
                .join( # Join back with the main pitches df
                    full_df.group_by('game_pk').first(),
                    on='game_pk',
                    how='left'
                )
                .select(['game_pk', 'play_type', 'temperature', 'in', 'out', 'rtl', 'ltr', 'game_play_share']) # Pluck only the weather columns
                .with_columns((pl.col('temperature') ** 2).alias('temperature_sq')) # And square temprature for feature engineering purposes
                .drop('temperature')
            )

            weather_regression_dfs[pitbat_combo] = weather_regression_df

        return weather_regression_dfs
    
     ######################################################################################
    def _compute_weather_regression_coefficients(self, weather_training_data):
        """
        Regresses the percent of plays in a game for each play type on the underlying weather conditions 
        to determine the impact of weather on the distribution of play types. The regression results 
        will be used to neutralize batting statistics for use in modeling.

        Parameters:
        --------------
        all_plays_by_hand_combo (dict): 
            A dictionary of DataFrames representing the un-neutralized set of plays, 
            segmented by batter-pitcher handedness combinations. Each DataFrame includes 
            play types, game details, and weather-related features.

        Returns:
        --------------
        weather_coefficients (dict):
            A nested dictionary containing the regression coefficients for each weather 
            datapoint (temperature, wind direction, and other weather factors) for each 
            play type within each batter-pitcher handedness combination. The structure is as follows:

            {
                "pitbat_combo_1": {
                    "play_type_1": {
                        "intercept": <intercept_value>,
                        "temprature_sq": <coefficient_value>,
                        "wind_ltr": <coefficient_value>,
                        "wind_rtl": <coefficient_value>,
                        "wind_in": <coefficient_value>,
                        "wind_out": <coefficient_value>
                    },
                    ...
                },
                "pitbat_combo_2": {
                    ...
                },
                ...
            }
        """

        for k in weather_training_data:
            weather_training_data[k] = weather_training_data[k].clone().collect().to_pandas()

        weather_coefficients = {}

        for pitbat_combo in constants.HAND_COMBOS:
            weather_coefficients[pitbat_combo] = {}

            # Segment to only the specific play type for each play type before regressing on the weather info
            for play_type in weather_training_data[pitbat_combo].play_type.unique():
                regression_df = weather_training_data[pitbat_combo][
                    weather_training_data[pitbat_combo].play_type == play_type]

                # Remove outliers for game_share_delta, most of which are caused by low pitbat_combo sample sizes in games. However only do so if there are non 'outliers'. The else
                # triggers is early in the season there is a play like int. walk that has not happened in a game and all game play shares are 0
                regression_df = regression_df[(np.abs(stats.zscore(regression_df.game_play_share)) < 3)] if len(
                    regression_df[(np.abs(stats.zscore(regression_df.game_play_share)) < 3)]) > 0 else regression_df

                # Create 2 sets of x data, with and without squaring temprature
                x_sq = regression_df[[
                    "temperature_sq", "ltr", "rtl", "in", "out"]].copy()
                scaler = StandardScaler()
                x_sq_scaled = scaler.fit_transform(x_sq)

                # Save the scaler in case we dont regress every day and need to reuse the 'training' one
                scaler_save_path = Path('data/weather_regression_scaler')
                scaler_save_path.mkdir(parents=True, exist_ok=True)
                with open(scaler_save_path, 'wb') as fpath:
                    pkl.dump(scaler, fpath)

                y = regression_df.game_play_share

                # Regress the temprature squared dataset on game_share_delta
                lin_sq = LinearRegression(fit_intercept=True)
                lin_sq.fit(x_sq_scaled, y)

                weather_coefficients[pitbat_combo][play_type] = {"intercept": lin_sq.intercept_, "temperature_sq": lin_sq.coef_[0], "wind_ltr": lin_sq.coef_[1],
                                                                 "wind_rtl": lin_sq.coef_[2], "wind_in": lin_sq.coef_[3], "wind_out": lin_sq.coef_[4]}

        return weather_coefficients

    def _compute_park_factors(self, all_plays_by_hand_combo):
        """
        Calculates the park factor for each ballpark and play type based on the relative frequency 
        of each play type occurring at the ballpark versus other parks.

        The park factor represents how much more or less likely a play type is to occur at a 
        specific ballpark compared to the league average.

        Parameters:
        --------------
        all_plays_by_hand_combo (dict): 
            A dictionary of DataFrames where each key represents a batter-pitcher handedness combination 
            and the associated DataFrame contains the un-neutralized set of plays. Each DataFrame includes 
            information about the play type, ballpark, and other relevant game data.

        Returns:
        --------------
        park_factors_dict (dict): 
            A nested dictionary containing the park factors for each ballpark and play type. 
            The structure is as follows:

            {
                "pitbat_combo_1": {
                    "ballpark_1": {
                        "play_type_1": <park_factor_value>,
                        "play_type_2": <park_factor_value>,
                        ...
                    },
                    "ballpark_2": {
                        ...
                    },
                    ...
                },
                "pitbat_combo_2": {
                    ...
                },
                ...
            }

            If a park factor cannot be computed (e.g., insufficient data), the value will be set to `"n/a"`.
        """

        park_factors_dict = {}

        if self.verbose:
            print("Calculating Ballpark Factors")

        for pitbat_combo in constants.HAND_COMBOS:
            park_factors_dict[pitbat_combo] = {}

            # Calculate the percentage of each play share for each ballpark
            in_ballpark = (all_plays_by_hand_combo[pitbat_combo]
                           .group_by(['Stadium', 'play_type'])
                           .agg(pl.len().alias('ballpark_count'))
            )

            in_ballpark_total = (all_plays_by_hand_combo[pitbat_combo]
                                 .group_by('Stadium')
                                 .agg(pl.len().alias('ballpark_total'))
            )

            ballpark_play_share = (in_ballpark.join(in_ballpark_total, on='Stadium', how='left')
                                   .with_columns((pl.col('ballpark_count') / pl.col('ballpark_total')).alias('in_park_rate'))
            )

            # Calculate the percentage of each play share outside of each park
            total_plays = (
                all_plays_by_hand_combo[pitbat_combo]
                .group_by('play_type')
                .agg(pl.len().alias('total_plays'))
            )

            # Total plays across the league
            out_ballpark = (
                in_ballpark.join(total_plays, how='left', on='play_type')
                .with_columns([(pl.col('total_plays') - pl.col('ballpark_count')).alias('out_of_park_count'),
                               (pl.sum('ballpark_count').alias('total_global_plays'))
                               ])
            )

            # Total plays that happened outside of the stadium
            out_ballpark = out_ballpark.with_columns([
                ((pl.col('total_global_plays') - pl.col('total_plays')).alias('total_out_of_park_plays'))
            ])

            # The out of park rate = out of park play type that happened / total plays outside of the stadium       
            out_ballpark = out_ballpark.with_columns([
                ((pl.col('out_of_park_count') / pl.col('total_out_of_park_plays')).alias('out_of_park_rate'))
            ]).drop('total_plays', 'out_of_park_count', 'total_global_plays')

            # Join the in and out of park DFs and calculate the specific park factors
            # Then convert to a dict
            ballpark_factors = (
                ballpark_play_share.join(out_ballpark, on=['Stadium', 'play_type'])
                .with_columns([(pl.col('in_park_rate') / pl.col('out_of_park_rate')).alias('park_factor')])
                .select(['Stadium', 'play_type', 'park_factor'])
            ).collect().to_pandas()

            ballpark_factors = {
                stadium: group.set_index("play_type")["park_factor"].to_dict()
                for stadium, group in ballpark_factors.groupby("Stadium")
            }

            park_factors_dict[pitbat_combo] = ballpark_factors

        return park_factors_dict
    ######################################################################################

    ######################################################################################
    ######################################################################################
    def build_neutralization_coefficient_dictionaries(self, all_plays_by_hand_combo):
        """
        Builds a dictionary containing the neutralization coefficients for weather conditions and park factors.
        The function combines weather regression coefficients and park factors to produce a dictionary 
        that can be used for neutralizing player performance based on external factors such as weather 
        and ballpark conditions.

        Parameters:
        --------------
        all_plays_by_hand_combo (dict): 
            A dictionary of DataFrames where each key represents a batter-pitcher handedness combination 
            and the associated DataFrame contains the un-neutralized set of plays. Each DataFrame includes 
            information about the play type, ballpark, weather conditions, and other relevant game data.

        Returns:
        --------------
        dict: 
            A dictionary containing two nested dictionaries:

            - "weather_coefficients": Contains the regression coefficients for weather-related factors
            (e.g., temperature, wind direction) affecting play type distributions for each batter-pitcher combo.

            - "park_factors": Contains the park factors for each ballpark and play type, indicating how much 
            more or less likely a play type is to occur in a specific ballpark compared to others.

            Example structure:
            {
                "weather_coefficients": {
                    "pitbat_combo_1": {
                        "play_type_1": {"intercept": <value>, "temprature_sq": <value>, ...},
                        "play_type_2": {...},
                        ...
                    },
                    "pitbat_combo_2": {...},
                    ...
                },
                "park_factors": {
                    "pitbat_combo_1": {
                        "ballpark_1": {"play_type_1": <park_factor_value>, ...},
                        "ballpark_2": {...},
                        ...
                    },
                    "pitbat_combo_2": {...},
                    ...
                }
            }
        """
        weather_coefficients = self._compute_weather_regression_coefficients(
            all_plays_by_hand_combo)
        park_factors = self._compute_park_factors(all_plays_by_hand_combo)

        return {"weather_coefficients": weather_coefficients, "park_factors": park_factors}
    ######################################################################################