import constants
import time
import pandas as pd
import numpy as np
import pickle as pkl
import sys
import json
import re

from scipy import stats
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import timedelta
from IPython.display import clear_output

from gcloud_helper import cloud_functions as cf # type: ignore




class dataset_builder():

    def __init__(self, rolling_windows=[75, 504], verbose=False):
        self.rolling_windows = rolling_windows
        self.verbose = verbose

    ######################################################################################
    # Clean Pitch Data
    ######################################################################################
    def _get_wind_direction(self, full_weather: str) -> str:
        """
        Extracts the wind direction from a full weather description scraped from a baseball 
        reference box score (e.g., https://www.baseball-reference.com/boxes/CHA/CHA202407080.shtml).

        Parameters:
            full_weather (str): The raw weather information string.

        Returns:
            str: A simplified wind direction, one of ["in", "out", "Right to Left", "Left to Right"], 
                or None if no direction is found.
        """
        if full_weather != None:
            if "in" in "".join(full_weather.split("Wind")) or "In" in "".join(full_weather.split("Wind")):
                weather = "in" #full_weather.full_weather.split("mph ")[-1].split(' from')[0]
            elif "out" in "".join(full_weather.split("Wind")) or "Out" in "".join(full_weather.split("Wind")):
                weather = "out" #full_weather.full_weather.split("mph ")[-1].split(' to')[0]
            elif "Left" in "".join(full_weather.split("Wind")) or "Right" in "".join(full_weather.split("Wind")):
                weather = full_weather.split("from ")[-1].strip(".").split(", ")[0]
            else: # Sometimes wind just listed as 0mph with no direction
                weather = None
        else:
            weather =  None

        return weather

    def _convert_wind_direction(self, all_plays_by_pitbat_combo, wind_column = "wind_direction"):
        """
        Converts categorical wind direction and numeric wind speed in a DataFrame into 
        one-hot encoded (OHE) columns representing wind speed in each direction.

        Parameters:
            all_plays_by_pitbat_combo (DataFrame): A cleaned DataFrame of pitches, including columns 
                for wind direction and wind speed of each play.
            wind_column (str): The column name containing wind direction data. Defaults to "wind_direction".

        Returns:
            DataFrame: The input DataFrame with added OHE columns for wind direction, where each column 
            represents a wind direction and contains the wind speed in that direction.
        """
        
        # When wind speed is 0, the direction is automatically listed as "in" --> convert it to "zero" to differentiate
        ind = all_plays_by_pitbat_combo[all_plays_by_pitbat_combo.wind_speed.values == 0].index
        all_plays_by_pitbat_combo.loc[ind, "wind_direction"] = "zero"
        
        # Use pd.get_dummies to One Hot Encode the wind direction as binary columns
        wind_columns = pd.get_dummies(wind_column, columns=['categorical_column', ])
        wind_columns = pd.concat([all_plays_by_pitbat_combo, wind_columns], axis = 1)
        
        # Finally multiply the binary wind direction columns by the wind speed to get the final wind speed in the correct direction
        for column in wind_columns.columns[-5:]:
            wind_columns[column] = wind_columns[column] * wind_columns["wind_speed"]
        
        return wind_columns

    def _pull_full_weather(self, game_date, home_team, away_team, total_weather_df):
        try:
            #value = total_weather_df[(total_weather_df.date.values == game_date) & ((total_weather_df.home_team.values == constants.WEATHER_NAME_CONVERSIONS[home_team])|(total_weather_df.home_team.values == constants.WEATHER_NAME_CONVERSIONS[away_team]))].weather.iloc[0]                                                                                       
            value = total_weather_df[(total_weather_df.date.values == game_date) & ((total_weather_df.converted_home_team.values == home_team)|(total_weather_df.converted_home_team.values == away_team))].weather.iloc[0]                                                                                       
            return value
        except:
            return 'Start Time Weather: 72° F, Wind 0mph, In Dome.'


    def _correct_home_away_swap(self, all_pitches: pd.DataFrame) -> pd.DataFrame:
        """Function used to correct a series of games across years where the home team and away team are swapped on baseball reference
        
        ------------INPUTS------------
        - all_pitches: DataFrame
            A dataframe of individual pitches, pulled from the statcast API.
            
        ------------OUTPUTS------------
        - all_pitches: DataFrame
            A dataframe of individual pitches, identical to the function's input, other than the correction of home and away teams in
            a select subset of games.

        """
        strange_games_I = all_pitches[(all_pitches.home_team == "TOR") & (all_pitches.away_team == "WSH")].index
        all_pitches.loc[strange_games_I, "home_team"] = "WSH"
        all_pitches.loc[strange_games_I, "away_team"] = "TOR"

        strange_games_II = all_pitches[(all_pitches.home_team == "CIN") & (all_pitches.away_team == "SF") & (all_pitches.game_date == "2013-07-23")].index
        all_pitches.loc[strange_games_II, "home_team"] = "SF"
        all_pitches.loc[strange_games_II, "away_team"] = "CIN"

        strange_games_III = all_pitches[(all_pitches.home_team == "BAL") & (all_pitches.away_team == "TB") & (all_pitches.game_date == "2015-05-01")].index
        all_pitches.loc[strange_games_III, "home_team"] = "TB"
        all_pitches.loc[strange_games_III, "away_team"] = "BAL"

        strange_games_IV = all_pitches[(all_pitches.home_team == "BAL") & (all_pitches.away_team == "TB") & (all_pitches.game_date == "2015-05-02")].index
        all_pitches.loc[strange_games_IV, "home_team"] = "TB"
        all_pitches.loc[strange_games_IV, "away_team"] = "BAL"

        strange_games_V = all_pitches[(all_pitches.home_team == "BAL") & (all_pitches.away_team == "TB") & (all_pitches.game_date == "2015-05-03")].index
        all_pitches.loc[strange_games_V, "home_team"] = "TB"
        all_pitches.loc[strange_games_V, "away_team"] = "BAL"

        strange_games_VI = all_pitches[(all_pitches.home_team == "MIA") & (all_pitches.away_team == "MIL") & (all_pitches.game_date == "2017-09-16")].index
        all_pitches.loc[strange_games_VI, "home_team"] = "MIL"
        all_pitches.loc[strange_games_VI, "away_team"] = "MIA"

        strange_games_VII = all_pitches[(all_pitches.home_team == "MIA") & (all_pitches.away_team == "MIL") & (all_pitches.game_date == "2017-09-17")].index
        all_pitches.loc[strange_games_VII, "home_team"] = "MIL"
        all_pitches.loc[strange_games_VII, "away_team"] = "MIA"

        strange_games_VIII = all_pitches[(all_pitches.home_team == "MIA") & (all_pitches.away_team == "MIL") & (all_pitches.game_date == "2017-09-15")].index
        all_pitches.loc[strange_games_VIII, "home_team"] = "MIL"
        all_pitches.loc[strange_games_VIII, "away_team"] = "MIA"

        strange_games_IX = all_pitches[(all_pitches.home_team == "NYY") & (all_pitches.away_team == "PHI") & (all_pitches.game_date == "2020-08-05")].index
        all_pitches.loc[strange_games_IX, "home_team"] = "PHI"
        all_pitches.loc[strange_games_IX, "away_team"] = "NYY"

        strange_games_X = all_pitches[(all_pitches.home_team == "MIA") & (all_pitches.away_team == "BAL") & (all_pitches.game_date == "2020-08-05")].index
        all_pitches.loc[strange_games_X, "home_team"] = "BAL"
        all_pitches.loc[strange_games_X, "away_team"] = "MIA"

        strange_games_XI = all_pitches[(all_pitches.home_team == "MIA") & (all_pitches.away_team == "BAL") & (all_pitches.game_date == "2020-08-06")].index
        all_pitches.loc[strange_games_XI, "home_team"] = "BAL"
        all_pitches.loc[strange_games_XI, "away_team"] = "MIA"

        strange_games_XII = all_pitches[(all_pitches.home_team == "MIA") & (all_pitches.away_team == "BAL") & (all_pitches.game_date == "2020-08-07")].index
        all_pitches.loc[strange_games_XII, "home_team"] = "BAL"
        all_pitches.loc[strange_games_XII, "away_team"] = "MIA"

        strange_games_XIII = all_pitches[(all_pitches.home_team == "STL") & (all_pitches.away_team == "CHC") & (all_pitches.game_date == "2020-08-17")].index
        all_pitches.loc[strange_games_XIII, "home_team"] = "CHC"
        all_pitches.loc[strange_games_XIII, "away_team"] = "STL"

        strange_games_XIX = all_pitches[(all_pitches.home_team == "STL") & (all_pitches.away_team == "CHC") & (all_pitches.game_date == "2020-08-18")].index
        all_pitches.loc[strange_games_XIX, "home_team"] = "CHC"
        all_pitches.loc[strange_games_XIX, "away_team"] = "STL"

        strange_games_XX = all_pitches[(all_pitches.home_team == "STL") & (all_pitches.away_team == "CHC") & (all_pitches.game_date == "2020-08-19")].index
        all_pitches.loc[strange_games_XX, "home_team"] = "CHC"
        all_pitches.loc[strange_games_XX, "away_team"] = "STL"

        strange_games_XXI = all_pitches[(all_pitches.home_team == "MIA") & (all_pitches.away_team == "WSH") & (all_pitches.game_date == "2020-08-22")].index
        all_pitches.loc[strange_games_XXI, "home_team"] = "WSH"
        all_pitches.loc[strange_games_XXI, "away_team"] = "MIA"

        strange_games_XXII = all_pitches[(all_pitches.home_team == "MIA") & (all_pitches.away_team == "NYM") & (all_pitches.game_date == "2020-08-25")].index
        all_pitches.loc[strange_games_XXII, "home_team"] = "NYM"
        all_pitches.loc[strange_games_XXII, "away_team"] = "MIA"

        strange_games_XXIII = all_pitches[(all_pitches.home_team == "NYY") & (all_pitches.away_team == "ATL") & (all_pitches.game_date == "2020-08-26")].index
        all_pitches.loc[strange_games_XXIII, "home_team"] = "ATL"
        all_pitches.loc[strange_games_XXIII, "away_team"] = "NYY"

        strange_games_XXIV = all_pitches[(all_pitches.home_team == "CIN") & (all_pitches.away_team == "MIL") & (all_pitches.game_date == "2020-08-27")].index
        all_pitches.loc[strange_games_XXIV, "home_team"] = "MIL"
        all_pitches.loc[strange_games_XXIV, "away_team"] = "CIN"

        strange_games_XXV = all_pitches[(all_pitches.home_team == "SEA") & (all_pitches.away_team == "SD") & (all_pitches.game_date == "2020-08-27")].index
        all_pitches.loc[strange_games_XXV, "home_team"] = "SD"
        all_pitches.loc[strange_games_XXV, "away_team"] = "SEA"

        strange_games_XXVI = all_pitches[(all_pitches.home_team == "LAD") & (all_pitches.away_team == "SF") & (all_pitches.game_date == "2020-08-27")].index
        all_pitches.loc[strange_games_XXVI, "home_team"] = "SF"
        all_pitches.loc[strange_games_XXVI, "away_team"] = "LAD"

        strange_games_XXVII = all_pitches[(all_pitches.home_team == "PIT") & (all_pitches.away_team == "STL") & (all_pitches.game_date == "2020-08-27")].index
        all_pitches.loc[strange_games_XXVII, "home_team"] = "STL"
        all_pitches.loc[strange_games_XXVII, "away_team"] = "PIT"

        strange_games_XXVIII = all_pitches[(all_pitches.home_team == "NYM") & (all_pitches.away_team == "NYY") & (all_pitches.game_date == "2020-08-28")].index
        all_pitches.loc[strange_games_XXVIII, "home_team"] = "NYY"
        all_pitches.loc[strange_games_XXVIII, "away_team"] = "NYM"

        strange_games_XXVIV = all_pitches[(all_pitches.home_team == "MIN") & (all_pitches.away_team == "DET") & (all_pitches.game_date == "2020-08-29")].index
        all_pitches.loc[strange_games_XXVIV, "home_team"] = "DET"
        all_pitches.loc[strange_games_XXVIV, "away_team"] = "MIN"

        strange_games_XXVV = all_pitches[(all_pitches.home_team == "OAK") & (all_pitches.away_team == "HOU") & (all_pitches.game_date == "2020-08-29")].index
        all_pitches.loc[strange_games_XXVV, "home_team"] = "HOU"
        all_pitches.loc[strange_games_XXVV, "away_team"] = "OAK"

        strange_games_XXVVI = all_pitches[(all_pitches.home_team == "CHC") & (all_pitches.away_team == "CIN") & (all_pitches.game_date == "2020-08-29")].index
        all_pitches.loc[strange_games_XXVVI, "home_team"] = "CIN"
        all_pitches.loc[strange_games_XXVVI, "away_team"] = "CHC"

        strange_games_XXVVII = all_pitches[(all_pitches.home_team == "NYM") & (all_pitches.away_team == "NYY") & (all_pitches.game_date == "2020-08-30")].index
        all_pitches.loc[strange_games_XXVVII, "home_team"] = "NYY"
        all_pitches.loc[strange_games_XXVVII, "away_team"] = "NYM"

        strange_games_XXVVIII = all_pitches[(all_pitches.home_team == "WSH") & (all_pitches.away_team == "ATL") & (all_pitches.game_date == "2020-09-4")].index
        all_pitches.loc[strange_games_XXVVIII, "home_team"] = "ATL"
        all_pitches.loc[strange_games_XXVVII, "away_team"] = "WSH"

        strange_games_XXVVIV = all_pitches[(all_pitches.home_team == "NYY") & (all_pitches.away_team == "BAL") & (all_pitches.game_date == "2020-09-04")].index
        all_pitches.loc[strange_games_XXVVIV, "home_team"] = "BAL"
        all_pitches.loc[strange_games_XXVVIV, "away_team"] = "NYY"

        strange_games_XXVVV = all_pitches[(all_pitches.home_team == "TOR") & (all_pitches.away_team == "BOS") & (all_pitches.game_date == "2020-09-04")].index
        all_pitches.loc[strange_games_XXVVV, "home_team"] = "BOS"
        all_pitches.loc[strange_games_XXVVV, "away_team"] = "TOR"

        strange_games_XXVVVI = all_pitches[(all_pitches.home_team == "DET") & (all_pitches.away_team == "MIN") & (all_pitches.game_date == "2020-09-04")].index
        all_pitches.loc[strange_games_XXVVVI, "home_team"] = "MIN"
        all_pitches.loc[strange_games_XXVVVI, "away_team"] = "DET"

        strange_games_XXVVVII = all_pitches[(all_pitches.home_team == "CIN") & (all_pitches.away_team == "PIT") & (all_pitches.game_date == "2020-09-04")].index
        all_pitches.loc[strange_games_XXVVVII, "home_team"] = "PIT"
        all_pitches.loc[strange_games_XXVVVII, "away_team"] = "CIN"

        strange_games_XXVVVIII = all_pitches[(all_pitches.home_team == "HOU") & (all_pitches.away_team == "LAA") & (all_pitches.game_date == "2020-09-05")].index
        all_pitches.loc[strange_games_XXVVVIII, "home_team"] = "LAA"
        all_pitches.loc[strange_games_XXVVVIII, "away_team"] = "HOU"

        strange_games_XXVVVIV = all_pitches[(all_pitches.home_team == "STL") & (all_pitches.away_team == "CHC") & (all_pitches.game_date == "2020-09-05")].index
        all_pitches.loc[strange_games_XXVVVIV, "home_team"] = "CHC"
        all_pitches.loc[strange_games_XXVVVIV, "away_team"] = "STL"

        strange_games_L = all_pitches[(all_pitches.home_team == "HOU") & (all_pitches.away_team == "OAK") & (all_pitches.game_date == "2020-09-08")].index
        all_pitches.loc[strange_games_L, "home_team"] = "OAK"
        all_pitches.loc[strange_games_L, "away_team"] = "HOU"

        strange_games_LI = all_pitches[(all_pitches.home_team == "BOS") & (all_pitches.away_team == "PHI") & (all_pitches.game_date == "2020-09-08")].index
        all_pitches.loc[strange_games_LI, "home_team"] = "PHI"
        all_pitches.loc[strange_games_LI, "away_team"] = "BOS"

        strange_games_LII = all_pitches[(all_pitches.home_team == "MIN") & (all_pitches.away_team == "STL") & (all_pitches.game_date == "2020-09-08")].index
        all_pitches.loc[strange_games_LII, "home_team"] = "STL"
        all_pitches.loc[strange_games_LII, "away_team"] = "MIN"

        strange_games_LIII = all_pitches[(all_pitches.home_team == "DET") & (all_pitches.away_team == "STL") & (all_pitches.game_date == "2020-09-10")].index
        all_pitches.loc[strange_games_LIII, "home_team"] = "STL"
        all_pitches.loc[strange_games_LIII, "away_team"] = "DET"

        strange_games_LIV = all_pitches[(all_pitches.home_team == "PHI") & (all_pitches.away_team == "MIA") & (all_pitches.game_date == "2020-09-11")].index
        all_pitches.loc[strange_games_LIV, "home_team"] = "MIA"
        all_pitches.loc[strange_games_LIV, "away_team"] = "PHI"

        strange_games_LV = all_pitches[(all_pitches.home_team == "BAL") & (all_pitches.away_team == "NYY") & (all_pitches.game_date == "2020-09-11")].index
        all_pitches.loc[strange_games_LV, "home_team"] = "NYY"
        all_pitches.loc[strange_games_LV, "away_team"] = "BAL"

        strange_games_LVI = all_pitches[(all_pitches.home_team == "OAK") & (all_pitches.away_team == "TEX") & (all_pitches.game_date == "2020-09-12")].index
        all_pitches.loc[strange_games_LVI, "home_team"] = "TEX"
        all_pitches.loc[strange_games_LVI, "away_team"] = "OAK"

        strange_games_LVII = all_pitches[(all_pitches.home_team == "PHI") & (all_pitches.away_team == "MIA") & (all_pitches.game_date == "2020-09-13")].index
        all_pitches.loc[strange_games_LVII, "home_team"] = "MIA"
        all_pitches.loc[strange_games_LVII, "away_team"] = "PHI"

        strange_games_LVIII = all_pitches[(all_pitches.home_team == "SF") & (all_pitches.away_team == "SD") & (all_pitches.game_date == "2020-09-13")].index
        all_pitches.loc[strange_games_LVII, "home_team"] = "SD"
        all_pitches.loc[strange_games_LVII, "away_team"] = "SF"

        strange_games_LIX = all_pitches[(all_pitches.home_team == "PIT") & (all_pitches.away_team == "CIN") & (all_pitches.game_date == "2020-09-14")].index
        all_pitches.loc[strange_games_LIX, "home_team"] = "CIN"
        all_pitches.loc[strange_games_LIX, "away_team"] = "PIT"

        strange_games_LX = all_pitches[(all_pitches.home_team == "STL") & (all_pitches.away_team == "MIL") & (all_pitches.game_date == "2020-09-14")].index
        all_pitches.loc[strange_games_LX, "home_team"] = "MIL"
        all_pitches.loc[strange_games_LX, "away_team"] = "STL"

        strange_games_LXI = all_pitches[(all_pitches.home_team == "OAK") & (all_pitches.away_team == "SEA") & (all_pitches.game_date == "2020-09-14")].index
        all_pitches.loc[strange_games_LXI, "home_team"] = "SEA"
        all_pitches.loc[strange_games_LXI, "away_team"] = "OAK"

        strange_games_LXII = all_pitches[(all_pitches.home_team == "STL") & (all_pitches.away_team == "MIL") & (all_pitches.game_date == "2020-09-16")].index
        all_pitches.loc[strange_games_LXII, "home_team"] = "MIL"
        all_pitches.loc[strange_games_LXII, "away_team"] = "STL"

        strange_games_LXIII = all_pitches[(all_pitches.home_team == "TB") & (all_pitches.away_team == "BAL") & (all_pitches.game_date == "2020-09-17")].index
        all_pitches.loc[strange_games_LXIII, "home_team"] = "BAL"
        all_pitches.loc[strange_games_LXIII, "away_team"] = "TB"

        strange_games_LXIV = all_pitches[(all_pitches.home_team == "WSH") & (all_pitches.away_team == "MIA") & (all_pitches.game_date == "2020-09-18")].index
        all_pitches.loc[strange_games_LXIV, "home_team"] = "MIA"
        all_pitches.loc[strange_games_LXIV, "away_team"] = "WSH"

        strange_games_LXV = all_pitches[(all_pitches.home_team == "TOR") & (all_pitches.away_team == "PHI") & (all_pitches.game_date == "2020-09-18")].index
        all_pitches.loc[strange_games_LXV, "home_team"] = "PHI"
        all_pitches.loc[strange_games_LXV, "away_team"] = "TOR"

        strange_games_LXVI = all_pitches[(all_pitches.home_team == "STL") & (all_pitches.away_team == "PIT") & (all_pitches.game_date == "2020-09-18")].index
        all_pitches.loc[strange_games_LXVI, "home_team"] = "PIT"
        all_pitches.loc[strange_games_LXVI, "away_team"] = "STL"

        strange_games_LXVII = all_pitches[(all_pitches.home_team == "WSH") & (all_pitches.away_team == "MIA") & (all_pitches.game_date == "2020-09-20")].index
        all_pitches.loc[strange_games_LXVII, "home_team"] = "MIA"
        all_pitches.loc[strange_games_LXVII, "away_team"] = "WSH"

        strange_games_LXVIII = all_pitches[(all_pitches.home_team == "PHI") & (all_pitches.away_team == "WSH") & (all_pitches.game_date == "2020-09-22")].index
        all_pitches.loc[strange_games_LXVIII, "home_team"] = "WSH"
        all_pitches.loc[strange_games_LXVIII, "away_team"] = "PHI"

        strange_games_LXIX = all_pitches[(all_pitches.home_team == "COL") & (all_pitches.away_team == "ARI") & (all_pitches.game_date == "2020-09-25")].index
        all_pitches.loc[strange_games_LXIX, "home_team"] = "ARI"
        all_pitches.loc[strange_games_LXIX, "away_team"] = "COL"

        strange_games_LXX = all_pitches[(all_pitches.home_team == "SD") & (all_pitches.away_team == "SF") & (all_pitches.game_date == "2020-09-25")].index
        all_pitches.loc[strange_games_LXX, "home_team"] = "SF"
        all_pitches.loc[strange_games_LXX, "away_team"] = "SD"

        strange_games_LXXI = all_pitches[(all_pitches.home_team == "MIL") & (all_pitches.away_team == "STL") & (all_pitches.game_date == "2020-09-25")].index
        all_pitches.loc[strange_games_LXXI, "home_team"] = "STL"
        all_pitches.loc[strange_games_LXXI, "away_team"] = "MIL"

        strange_games_LXXII = all_pitches[(all_pitches.home_team == "SEA") & (all_pitches.away_team == "OAK") & (all_pitches.game_date == "2020-09-26")].index
        all_pitches.loc[strange_games_LXXII, "home_team"] = "OAK"
        all_pitches.loc[strange_games_LXXII, "away_team"] = "SEA"

        strange_games_LXXIII = all_pitches[(all_pitches.home_team == "NYM") & (all_pitches.away_team == "WSH") & (all_pitches.game_date == "2020-09-26")].index
        all_pitches.loc[strange_games_LXXIII, "home_team"] = "WSH"
        all_pitches.loc[strange_games_LXXIII, "away_team"] = "NYM"

        strange_games_LXXIV = all_pitches[(all_pitches.home_team == "SEA") & (all_pitches.away_team == "SF") & (all_pitches.game_date == "2020-09-16")].index
        all_pitches.loc[strange_games_LXXIV, "home_team"] = "SF"
        all_pitches.loc[strange_games_LXXIV, "away_team"] = "SEA"

        strange_games_LXXV = all_pitches[(all_pitches.home_team == "SEA") & (all_pitches.away_team == "SF") & (all_pitches.game_date == "2020-09-17")].index
        all_pitches.loc[strange_games_LXXV, "home_team"] = "SF"
        all_pitches.loc[strange_games_LXXV, "away_team"] = "SEA"

        strange_games_LXXVI = all_pitches[(all_pitches.home_team == "SEA") & (all_pitches.away_team == "SD") & (all_pitches.game_date == "2020-09-18")].index
        all_pitches.loc[strange_games_LXXVI, "home_team"] = "SD"
        all_pitches.loc[strange_games_LXXVI, "away_team"] = "SEA"

        strange_games_LXXVII = all_pitches[(all_pitches.home_team == "SEA") & (all_pitches.away_team == "SD") & (all_pitches.game_date == "2020-09-19")].index
        all_pitches.loc[strange_games_LXXVII, "home_team"] = "SD"
        all_pitches.loc[strange_games_LXXVII, "away_team"] = "SEA"

        strange_games_LXXVIII = all_pitches[(all_pitches.home_team == "SEA") & (all_pitches.away_team == "SD") & (all_pitches.game_date == "2020-09-20")].index
        all_pitches.loc[strange_games_LXXVIII, "home_team"] = "SD"
        all_pitches.loc[strange_games_LXXVIII, "away_team"] = "SEA"

        strange_games_LXXXI = all_pitches[(all_pitches.home_team == "WSH") & (all_pitches.away_team == "TOR") & ((all_pitches.game_date == "2021-04-27")|(all_pitches.game_date == "2021-04-27"))].index
        all_pitches.loc[strange_games_LXXXI, "home_team"] = "TOR"
        all_pitches.loc[strange_games_LXXXI, "away_team"] = "WSH"

        strange_games_LXXIX = all_pitches[(all_pitches.home_team == "TOR") & (all_pitches.away_team == "LAA") & (all_pitches.game_date == "2021-08-10")].index
        all_pitches.loc[strange_games_LXXIX, "home_team"] = "LAA"
        all_pitches.loc[strange_games_LXXIX, "away_team"] = "TOR"

        strange_games_LXXX = all_pitches[(all_pitches.home_team == "OAK") & (all_pitches.away_team == "DET") & (all_pitches.game_date == "2022-05-10")].index
        all_pitches.loc[strange_games_LXXX, "home_team"] = "DET"
        all_pitches.loc[strange_games_LXXX, "away_team"] = "OAK"

        return all_pitches

    def _segregate_plays_by_pitbat_combo(self, cleaned_plays: pd.DataFrame) -> pd.DataFrame:
        """
        Segregates a DataFrame of cleaned plays into a dictionary, splitting the plays by 
        the four possible batter-pitcher handedness combinations.

        Parameters:
            cleaned_plays (DataFrame): A DataFrame of cleaned plays, typically the output of the 
                `clean_raw_pitches` function.

        Returns:
            dict: A dictionary with keys ["RR", "RL", "LR", "LL"], where each key corresponds to 
            a batter-pitcher handedness combo, and the value is a DataFrame of plays for that combo.
        """
        # Set up a Dictionary to hold all pitches, divided by the pitbat combo
        all_plays_by_pitbat_combo = {"RR":{}, "RL":{}, "LR":{}, "LL":{}}

        # Split all plays on combo of batter/pitcher handedness before placing into the dictionary
        for pitbat_combo in all_plays_by_pitbat_combo.keys(): 
            pitbat_df = cleaned_plays[(cleaned_plays.stand == pitbat_combo[0]) & (cleaned_plays.p_throws == pitbat_combo[1])].copy().reset_index(drop=True)
            all_plays_by_pitbat_combo[pitbat_combo]= pitbat_df

        return all_plays_by_pitbat_combo

    ################################################################################
    def clean_raw_pitches(self, raw_pitches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans a DataFrame of raw pitch data, filtering and transforming it into a usable format 
        for subsequent analyses, including attaching weather and ballpark information.

        Parameters:
            raw_pitches_df (DataFrame): A DataFrame of uncleaned pitch data from the Statcast API.

        Returns:
            dict: A dictionary with 4 keys ("RR", "RL", "LR", "LL"), each containing a DataFrame 
            of pitches divided by batter-pitcher handedness combination.
        """
        
        if self.verbose:
            print("Cleaning Data")

        # Filter down to only regular season games
        raw_pitches_df = raw_pitches_df[raw_pitches_df.game_type == "R"]

        # Correct home and away mistakes in the pitch data
        raw_pitches_df = self._correct_home_away_swap(raw_pitches_df)

        # Convert the datetime game_date to a string formatted as YYYY-MM-DD, and sort the df on the column to make sure everything is in order
        raw_pitches_df.game_date = raw_pitches_df.game_date.apply(lambda x: str(x).split(" ")[0])
        raw_pitches_df = raw_pitches_df.sort_values(by = ["game_date", "inning", "inning_topbot", "at_bat_number"], ascending = True)

        # Filter all pitches to only those with an event
        raw_plays = raw_pitches_df[pd.isna(raw_pitches_df.events) == False]

        # Filter all pitches with an event to only those types we care about
        relevant_plays = raw_plays[raw_plays.events.isin(constants.RELEVANT_PLAY_TYPES)]

        # Sort Plays by Date, Inning, Top/Bot of Inning, and if possible batter number of inning to ensure all pitches/plays are in chronelogical order
        final_plays = relevant_plays[constants.RELEVANT_BATTING_COLUMNS].sort_values(by = "game_date").reset_index(drop = True)

        # Add a new column that groups all the event types into eventual Y labels
        final_plays["play_type"] = final_plays.events.apply(lambda x: constants.PLAY_TYPE_DICT[x])

        # Insert a new 'type counter' coulumn that will be used repeatedly for calculating rolling stats
        final_plays["type_counter"] = 1
        
        ############ ATTATCH WEATHER INFORMATION TO EACH PITCH ############

        # Build a combined weather dataframe by concatanating all yearly weather dataframes belonging to years present in the plays dataframe
        years = list(final_plays.game_date.apply(lambda x: x.split("-")[0]).unique())\

        weather_dictionary_holder = {}

        for year in years:
            yearly_weather_df = cf.CloudHelper().download_from_cloud("proreference_weather_data/weather_data_{}".format(year))# if this fails we might not have that years weather in cloud storage
            weather_dictionary_holder[year] = yearly_weather_df
        total_weather_df = pd.concat([df for df in weather_dictionary_holder.values()])

        # Create a new column with the converted home team names in total_weather_df
        total_weather_df['converted_home_team'] = total_weather_df['home_team'].map({v: k for k, v in constants.WEATHER_NAME_CONVERSIONS.items()})
        # Similarly, create a new column for the away team if necessary
        total_weather_df['converted_away_team'] = total_weather_df['away_team'].map({v: k for k, v in constants.WEATHER_NAME_CONVERSIONS.items()})
        
        # Drop the old columns
        total_weather_df = total_weather_df.drop(columns=['home_team', 'away_team'])

        # Attatch the raw weather string to the the play by matching the date and home team -- we've also added the away team clause to not throw errors on the limited number of cames coded wrong where in pitches data the teams didn't play eachother and were both on the road
        final_plays["full_weather"] = final_plays.apply(lambda x: self._pull_full_weather(x.game_date, x.home_team, x.away_team, total_weather_df), axis = 1)  
        
        # Break up the full weather info into temp, wind speed, and wind direction seperately
        final_plays["temprature"] = final_plays.full_weather.apply(lambda x: int(x.split(": ")[1].split("°")[0]))
        final_plays["wind_speed"] = final_plays.full_weather.apply(lambda x: int(x.split("Wind ")[1].split("mph")[0]) if "Wind" in x else 0)
        final_plays["wind_direction"] = final_plays.full_weather.apply(self._get_wind_direction)
        final_plays["wind_direction"] = final_plays.wind_direction.apply(lambda x: x.split(", ")[0] if x != None else x)
        
        # Convert the wind direction text column into a one-hot encoded set of columns multiplied by the wind speed (yields individual columns representing total wind speed)
        final_plays = self._convert_wind_direction(final_plays, final_plays.wind_direction)

        ############ ATTATCH BALLPARK INFO TO EACH PITCH ############

        # Import file to help connect team and year with a specific ballpark
        ballpark_info = pd.read_excel("Data/non_mlb_data/Ballpark Info.xlsx", header=2)[["Stadium", "Team", "Start Date", "End Date"]]

        # Create a column for the ballpark based on the date and home_team of each pitch
        final_plays["ballpark"] = final_plays.apply(lambda x: ballpark_info[(ballpark_info.Team.values == x.home_team) & (ballpark_info["End Date"].values > int(x.game_date.split("-")[0]))].Stadium.iloc[0],axis=1)

        ############ Divide pitches by pitbat combos in 4 dataframes ############
        all_plays_by_pitbat_combo = self._segregate_plays_by_pitbat_combo(final_plays)

        return all_plays_by_pitbat_combo

    #####################################################################################

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

        game_play_shares = {x:{} for x in constants.HAND_COMBOS}
        n = 0

        # Build a dictionary of all the game play shares for quicker insertion later
        for pitbat_combo in constants.HAND_COMBOS:
            full_df = all_plays_by_pitbat_combo[pitbat_combo].copy()
            # For each game
            for game in full_df.game_pk.unique(): #this might be able to become a groupby
                clear_output(wait = True)
                
                # Slice all pitches to just the individual game
                game_df = full_df[full_df.game_pk.values == game].copy()
                total_plays = len(game_df)
                
                # Calculate the total number of the play in the specific game by rolling 'type counter' within each play type and finding the max
                game_df["type_counter"] = game_df[["play_type", "game_pk", "type_counter"]].groupby(by = "play_type").cumsum().type_counter
                game_df = game_df[["play_type", "game_pk", "type_counter"]].groupby(by = "play_type").max()
                
                # Calculate the play share for each play type within the specific game by dividing the rolled counter by the total plays in the game
                game_df["play_share"]  = game_df.type_counter/total_plays 

                # Insert the game play shares for the specific game into a larger dictionary holder for later reference
                game_play_shares[pitbat_combo][game] = game_df

                # Update the counter and reprint to inform user of the current position
                if self.verbose:
                    if n%10000 == 0:
                        print("Calculating The Play Share by Play Type for Each Game. There are {}K Instances Remaining".format(round((sum([len(all_plays_by_pitbat_combo[x].game_pk.unique()) for x in constants.HAND_COMBOS])-n)/1000),6))
                    n+= 1
                clear_output(wait = True)
        
        plays_by_pitbat_combo_with_play_shares = {}
        # Add a column in the all plays dfs that is the game play share for the specific game and play type of each play
        for pitbat_combo in constants.HAND_COMBOS:
            if self.verbose:
                print("Inserting Play Shares by Play Type from Each Game To the All Pitches Data Set. There are {} Pitbat Combos Remaining".format(len(constants.HAND_COMBOS) - constants.HAND_COMBOS.index(pitbat_combo)))
                clear_output(wait = True)
            
            # The if statement in the apply below is used to catch the rare case (n=2 PA in 2018-2019) where the game_pk = <NA>. When this happens the play associated is in the game itself, but does not make it into the 
            # game_play_shares dict which throws and error when pulling the play type from the dictionary
            game_play_df = all_plays_by_pitbat_combo[pitbat_combo].copy()
            game_play_df["game_play_share"] = game_play_df.apply(lambda x: game_play_shares[pitbat_combo][x.game_pk].loc[x.play_type].play_share if x.play_type in game_play_shares[pitbat_combo][x.game_pk].index else 0, axis = 1)

            plays_by_pitbat_combo_with_play_shares[pitbat_combo] = game_play_df

        return plays_by_pitbat_combo_with_play_shares

    def _insert_missing_game_play_shares(self, weather_regression_data: dict, hand_combos: list = constants.HAND_COMBOS) -> dict:
        """
        Fills in missing play types for each game in the input dictionary by adding rows with a 
        `game_play_share` of 0 for missing play types in each game.

        Parameters:
            weather_regression_data (dict): A dictionary of DataFrames, each containing plays divided by 
                batter-pitcher handedness combo, including columns for play type, game identifier, and weather-related features.
            hand_combos (list): A list of batter-pitcher handedness combinations to iterate over (default is `constants.HAND_COMBOS`).

        Returns:
            dict: The updated dictionary of DataFrames, now with missing play types filled in for each game 
                with a `game_play_share` of 0.
        """

        # As the only plays in our data are types that happened in games, fill in all the missing play types for each game with a game_share of 0 for that play type
        play_types = constants.PLAY_TYPES
        n = 0
        for pitbat_combo in constants.HAND_COMBOS:
            for game in weather_regression_data[pitbat_combo].game_pk.unique():
                n += 1
                if self.verbose:
                    if n%10000 == 0:
                        print("Filling in the values for the game_play_share variable for games without the play (0). There are {}K Instances Remaining".format(round((sum([len(weather_regression_data[x].game_pk.unique()) for x in constants.HAND_COMBOS])-n)/1000),6))
                    clear_output(wait = True)
                    
                # Slice all plays to a specific game
                df = weather_regression_data[pitbat_combo][weather_regression_data[pitbat_combo].game_pk.values == game].copy()
                if len(df) < len(play_types): # Check if there are any missing plays and if so, determine how many and which ones
                    missing_plays = [play for play in play_types if play not in df.play_type.values]
                    num_missing_plays = len(missing_plays)
                    
                    # Pull all the game info for easy reference while inserting
                    game_info = df.iloc[0]
                    
                    # Build and insert into all pitches a DataFrame of each missing play from each game with the basic game info for the weather regression, including a game play share of 0
                    weather_regression_data[pitbat_combo] =  pd.concat([weather_regression_data[pitbat_combo], pd.DataFrame({"game_pk":[game]*num_missing_plays,
                                                                                                                            "game_date":[game_info.game_date]*num_missing_plays,
                                                                                                                            "play_type":missing_plays,
                                                                                                                            "temprature":[game_info.temprature]*num_missing_plays,
                                                                                                                            "Right to Left":[game_info["Right to Left"]]*num_missing_plays,
                                                                                                                            "Left to Right":[game_info["Left to Right"]]*num_missing_plays,
                                                                                                                            "in":[game_info["in"]]*num_missing_plays,
                                                                                                                            "out":[game_info["out"]]*num_missing_plays,
                                                                                                                            "zero":[game_info["zero"]]*num_missing_plays,
                                                                                                                            "game_play_share":[0]*num_missing_plays})])

        return weather_regression_data

    def _create_weather_regression_dataframes(self, all_plays_by_hand_combo):
        """
        Prepares data for weather-based regression models by calculating game play shares, 
        handling missing play types, and formatting data for each batter-pitcher handedness 
        combination (pitbat combo). The function processes and filters the necessary columns 
        for weather-related features and performs transformations like squaring the temperature.

        Parameters:
            all_plays_by_hand_combo (dict): A dictionary containing DataFrames of cleaned pitch data, 
                divided by batter-pitcher handedness combinations. Each DataFrame includes play type, 
                game details, and weather-related features.

        Returns:
            dict: A dictionary with weather training data, structured by batter-pitcher handedness combinations.
                Each entry contains a DataFrame of cleaned, transformed data ready for weather regression analysis.
        """

        # Start by filling in all the game play shares
        games_df = self._insert_game_play_shares(all_plays_by_hand_combo.copy())
        games_df = self._insert_missing_game_play_shares(games_df)

        weather_training_data = {x:{} for x in constants.HAND_COMBOS}
        l =  []

        # Clean the data to fit what we will need for weather regressions
        for pitbat_combo in constants.HAND_COMBOS:  
            weather_training_df = games_df[pitbat_combo].copy()
            
            # Remove any games with a month lower than 5 (May)
            weather_training_df = weather_training_df[weather_training_df.game_date.apply(lambda x: int(x.split("-")[1])) >=5]
            
            # Filter to only the columns we will need for the weather regressions
            weather_training_data[pitbat_combo] = weather_training_df[["game_pk", "play_type", "temprature", "Left to Right", "Right to Left",
                                                                    "in", "out", "zero", "game_play_share"]]
            
            # Square temprature to use in the regression because I believe it behaves this way
            weather_training_data[pitbat_combo]["temprature_squared"] = weather_training_data[pitbat_combo]["temprature"].apply(lambda x: x**2)

            # Group the weather training data by game and play type to get the game_play_share for each play type for each game
            weather_training_data[pitbat_combo] = weather_training_data[pitbat_combo].groupby(by = ["game_pk", "play_type"]).last().reset_index()
        
        clear_output(wait = False)
            
            
        return weather_training_data

    ######################################################################################
    def _compute_weather_regression_coefficients(self, all_plays_by_hand_combo):
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

        weather_training_data = self._create_weather_regression_dataframes(all_plays_by_hand_combo)

        weather_coefficients = {}

        for pitbat_combo in constants.HAND_COMBOS:
            weather_coefficients[pitbat_combo] = {}
            
            # Segment to only the specific play type for each play type before regressing on the weather info
            for play_type in weather_training_data[pitbat_combo].play_type.unique():
                regression_df = weather_training_data[pitbat_combo][weather_training_data[pitbat_combo].play_type == play_type]

                # Remove outliers for game_share_delta, most of which are caused by low pitbat_combo sample sizes in games. However only do so if there are non 'outliers'. The else
                # triggers is early in the season there is a play like int. walk that has not happened in a game and all game play shares are 0
                regression_df = regression_df[(np.abs(stats.zscore(regression_df.game_play_share)) < 3)] if len(regression_df[(np.abs(stats.zscore(regression_df.game_play_share)) < 3)]) >0 else regression_df
                
                # Create 2 sets of x data, with and without squaring temprature
                x = regression_df[["temprature", "Left to Right", "Right to Left", "in", "out", "zero"]].copy()
                x_sq = regression_df[["temprature_squared", "Left to Right", "Right to Left", "in", "out", "zero"]].copy()

                y = regression_df.game_play_share

                # Regress the temprature squared dataset on game_share_delta
                lin_sq = LinearRegression(fit_intercept = True)
                lin_sq.fit(x_sq, y)

                weather_coefficients[pitbat_combo][play_type] = {"intercept":lin_sq.intercept_, "temprature_sq":lin_sq.coef_[0], "wind_ltr":lin_sq.coef_[1],
                                                        "wind_rtl":lin_sq.coef_[2], "wind_in":lin_sq.coef_[3], "wind_out":lin_sq.coef_[4]}
                
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
            
            # For each ballpark, segment all our plays into 2 DataFrames. 1 for all plays at the park and 1 or all plays not at the park
            for ballpark in all_plays_by_hand_combo[pitbat_combo].ballpark.unique():
                park_factors_dict[pitbat_combo][ballpark] = {}
                at_park_df = all_plays_by_hand_combo[pitbat_combo][(all_plays_by_hand_combo[pitbat_combo].ballpark == ballpark)].copy()
                not_at_park_df = all_plays_by_hand_combo[pitbat_combo][(all_plays_by_hand_combo[pitbat_combo].ballpark != ballpark)].copy()

                # For each play type, calculate the percentage it occurs at in the park and out of the park
                for play_type in all_plays_by_hand_combo[pitbat_combo].play_type.unique():
                    at_park_rate = len(at_park_df[at_park_df.play_type == play_type])/len(at_park_df)
                    not_at_park_rate = len(not_at_park_df[not_at_park_df.play_type == play_type])/len(not_at_park_df)

                    try:
                        park_factor = at_park_rate/not_at_park_rate
                    except:
                        park_factor = "n/a"
                    
                    # Insert the park factors into a dictionary
                    park_factors_dict[pitbat_combo][ballpark][play_type] = park_factor

        clear_output(wait=False)
        
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
        weather_coefficients = self._compute_weather_regression_coefficients(all_plays_by_hand_combo)
        park_factors = self._compute_park_factors(all_plays_by_hand_combo)

        return {"weather_coefficients":weather_coefficients, "park_factors":park_factors}
    ######################################################################################
    ######################################################################################



    ############################## NEUTRALIZE ALL DATA #############################################

    def neutralize_stats(self, all_plays_by_hand_combo, coef_dict):
        """
        Neutralizes batting stats by adjusting for the impact of weather and stadium conditions. 
        This function applies the weather coefficients and park factors to each play based on its 
        actual weather information (temperature, wind speed/direction) and ballpark, calculating 
        an "impact" value for each individual play.

        Parameters:
        --------------
        all_plays_by_hand_combo (dict): 
            A dictionary of DataFrames, where each key represents a batter-pitcher handedness combination 
            and the associated DataFrame contains play data, including game-specific details like ballpark 
            and weather information (temperature, wind speed/direction).

        coef_dict (dict): 
            A nested dictionary containing two elements:
            - "weather_coefficients": Regression coefficients for weather-related factors (e.g., temperature, wind) affecting play type distributions.
            - "park_factors": Park factors for each ballpark and play type, indicating how much more or less likely a play type is to occur in a given ballpark.

        -----------------
        Returns: Tuple (DataFrame, Dictionary)
        --------------
        - DataFrame: The original `all_plays_by_hand_combo` DataFrame with added "impact" and "play_value" columns for each play.
        - Dictionary: A dictionary of factored training stats for each batter-pitcher combo, including neutralized batting stats.

        The function computes and adds the following columns to the DataFrame for each play:
        - "weather_expectation": Expected impact of weather on the play, calculated based on the weather coefficients.
        - "neutral_weather_expectation": The expected weather impact under neutral conditions (e.g., 72°F).
        - "weather_impact": Ratio of the actual weather impact to the neutral weather expectation.
        - "stadium_impact": Impact of the ballpark on the play, using park factors.
        - "impact": Total impact for each play, combining weather and stadium impacts.
        - "play_value": Adjusted value of the play, inversely proportional to the impact.

        Example structure of the returned dictionary:
        {
            "pitbat_combo_1": {
                "game_pk": [list of game IDs],
                "game_date": [list of game dates],
                ...
                "impact": [calculated impact values],
                "play_value": [calculated play values],
                ...
            },
            "pitbat_combo_2": { ... }
            ...
        }

        Example of usage:
        - Adjust batting stats based on external factors (weather, stadium) for better predictive modeling.
        """
        
        if self.verbose:
            print("Neutralizing Batting Stats using Weather/Stadium Coefficients")

        # Pull the corfficients dictionaries out from the combined dict in the function input
        weather_coefficients = coef_dict['weather_coefficients']
        park_factors_dict = coef_dict['park_factors']

        factored_training_stats = {}
        for pitbat_combo in constants.HAND_COMBOS:

            # Grab the relevant columns and games
            df = all_plays_by_hand_combo[pitbat_combo][["game_pk", "game_date", "batter", "pitcher",'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning',
                                                        'inning_topbot', "bat_score", "fld_score", "play_type", "temprature", "wind_speed", "wind_direction", "ballpark"]].copy()

            # Add information for the actual weather and stadium impacts for each game
            df = self._convert_wind_direction(df, df.wind_direction)
            df["weather_expectation"] = df.apply(lambda x: x["Left to Right"]*weather_coefficients[pitbat_combo][x.play_type]["wind_ltr"] + x["Right to Left"]*weather_coefficients[pitbat_combo][x.play_type]["wind_rtl"] +
                                            x["in"]*weather_coefficients[pitbat_combo][x.play_type]["wind_in"] + x["out"]*weather_coefficients[pitbat_combo][x.play_type]["wind_out"] +
                                            (x["temprature"]**2) * weather_coefficients[pitbat_combo][x.play_type]["temprature_sq"] + weather_coefficients[pitbat_combo][x.play_type]["intercept"], axis=1)

            df["neutral_weather_expectation"] = df.apply(lambda x: 72**2 * weather_coefficients[pitbat_combo][x.play_type]["temprature_sq"] + weather_coefficients[pitbat_combo][x.play_type]["intercept"], axis=1)
            df["weather_impact"] = df.weather_expectation/df.neutral_weather_expectation
            df["stadium_impact"] = df.apply(lambda x: park_factors_dict[pitbat_combo][x.ballpark][x.play_type], axis=1) # If delving further into project, we are technically doubling counting some of the weather impact in the stadium

            # Multiply the weather and stadium impacts to get the total impact for the specific at-bat result
            df["play_value"] = 1
            df["impact"] = df.play_value * df.weather_impact * df.stadium_impact
            df.play_value = 1/df.impact
            
            # Grab the final df that we will use for rolling stats
            factored_training_stats[pitbat_combo] = df[["game_pk", "game_date","ballpark", "temprature", "wind_speed", "wind_direction", "batter", "pitcher",
                                                        'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot', "bat_score", "fld_score", "play_type","impact", "play_value"]]
            
            # And finally sort so everything is in order
            factored_training_stats[pitbat_combo] = factored_training_stats[pitbat_combo].sort_values(by = ["game_date", "inning", "inning_topbot", 'outs_when_up'], ascending = True)

        clear_output(wait=False)
        
        return factored_training_stats
    ######################################################################################

    def roll_neutralized_batting_stats(self, neutralized_stats):
        """
        Rolls neutralized batting and pitching statistics across the tracked play types, aggregating the stats 
        on both a season and monthly basis, and ensures that the stats sum to 1 for each play type. 

        This function takes the output of `neutralize_stats`, which contains adjusted batting and pitching 
        stats with weather and park factors, and rolls the stats using a window-based calculation for both 
        batting and pitching. It calculates aggregated statistics for each player and then normalizes them 
        to ensure the percentages sum to 1. 

        Parameters:
        --------------
        neutralized_stats (dict of DataFrames): 
            A dictionary where the keys represent batter-pitcher handedness combinations (`pitbat_combo`), 
            and the values are DataFrames containing individual play data, including weather and park adjustments. 
            This is the output of the `neutralize_stats` function.

        -----------------
        Returns: 
        --------------
        dict: A dictionary containing two elements:
            - "pitching_stats": A dictionary of rolled pitching statistics for each batter-pitcher combo, including 
            season and monthly rolling stats for each play type.
            - "batting_stats": A dictionary of rolled batting statistics for each batter-pitcher combo, including 
            season and monthly rolling stats for each play type.

        The returned dictionaries are structured as follows:
        {
            "batting_stats": {
                "pitbat_combo_1": {
                    "game_pk": [list of game IDs],
                    "game_date": [list of game dates],
                    ...
                    "season_{play_type}": [season rolling stats for play type],
                    "month_{play_type}": [monthly rolling stats for play type],
                    ...
                },
                ...
            },
            "pitching_stats": {
                "pitbat_combo_1": { ... },
                ...
            }
        }

        Example of usage:
        - Roll batting stats for players to aggregate seasonal and monthly performance metrics.
        """
        rolling_windows = self.rolling_windows

        # Create a rolling percentage for each play outcome for each batter and pitcher for each year 
        rolling_factored_batting_stats = {}
        rolling_factored_pitching_stats = {}

        for pitbat_combo in constants.HAND_COMBOS:

            # Set up dictionaries to house everything
            rolling_factored_batting_stats[pitbat_combo] = {}
            rolling_factored_pitching_stats[pitbat_combo] = {}

            # Filter down to the stats for just the relevant hand combo and sort by game date for rolling
            batter_df, pitcher_df = neutralized_stats[pitbat_combo].copy(), neutralized_stats[pitbat_combo].copy()
            #batter_df, pitcher_df = batter_df.sort_values(by = "game_date", ascending = True), pitcher_df.sort_values(by = "game_date", ascending = True)
            batter_df["pitbat"] = pitbat_combo
            pitcher_df["pitbat"] = pitbat_combo
        
        ######################### APPLY NEUTRALIZED FACTORS TO UNDERLYING STATS #########################

            for play in constants.PLAY_TYPES:
                for rolling_window_length in rolling_windows:
                    batter_df[f"{rolling_window_length}_PA_{play}"] = (batter_df['play_type'] == play) * batter_df['play_value']
                    pitcher_df[f"{rolling_window_length}_PA_{play}"] = (pitcher_df['play_type'] == play) * pitcher_df['play_value']

                # # Vectorized calculation for batter_df
                # batter_df[f"season_{play}"] = (batter_df['play_type'] == play) * batter_df['play_value']
                # batter_df[f"month_{play}"] = batter_df[f"season_{play}"]
                
                # # Vectorized calculation for pitcher_df
                # pitcher_df[f"season_{play}"] = (pitcher_df['play_type'] == play) * pitcher_df['play_value']
                # pitcher_df[f"month_{play}"] = pitcher_df[f"season_{play}"]

        ######################### ROLL NEUTRALIZED STATS #########################
            for rolling_window_length in rolling_windows:
                rolling_batter_df = batter_df[['batter'] + [col for col in batter_df if f"{rolling_window_length}_PA" in col]].copy().groupby(by="batter").rolling(window=rolling_window_length, closed="left", min_periods=0).sum().reset_index()
                rolling_pitcher_df = pitcher_df[['pitcher'] + [col for col in pitcher_df if f"{rolling_window_length}_PA" in col]].copy().groupby(by="pitcher").rolling(window=rolling_window_length, closed="left", min_periods=0).sum().reset_index()

            for play in constants.PLAY_TYPES:
                batter_df[f"{rolling_window_length}_PA_{play}"] = rolling_batter_df[f"{rolling_window_length}_PA_{play}"]    
                pitcher_df[f"{rolling_window_length}_PA_{play}"] = rolling_pitcher_df[f"{rolling_window_length}_PA_{play}"]

        ######################### REPERCENTAGE NEUTRALIZED STATS (TO SUM % TO 1.0) #########################

            for rolling_window_length in rolling_windows:
                # Repercentage factored batting stats percentage to sum to 1 because they don't necessarily after neutralization
                cols = [f"{rolling_window_length}_PA_{play}" for play in constants.PLAY_TYPES]

                batter_df[cols] = batter_df[cols].div(batter_df[cols].sum(axis=1), axis=0)
                pitcher_df[cols] = pitcher_df[cols].div(pitcher_df[cols].sum(axis=1), axis=0)


            # # Roll batting stats on a season and monthly basis, and convert to a dict for speed later. Additionally, the closed input offsets the data by 1 row down, so that the values represent the percentages INCOMING to the plate appearence
            # season_rolled_batter_df = batter_df[['batter'] + [col for col in batter_df if "season_" in col]].copy().groupby(by="batter").rolling(window=504, closed="left", min_periods=0).sum().reset_index()
            # month_rolled_batter_df = batter_df[['batter'] + [col for col in batter_df if "month_" in col]].copy().groupby(by="batter").rolling(window=75, closed="left", min_periods=0).sum().reset_index()

            # # Roll pitching stats on a season and montly basis and convert to a dict for speed
            # season_rolled_pitcher_df = pitcher_df[['pitcher'] + [col for col in pitcher_df if "season_" in col]].copy().groupby(by="pitcher").rolling(window=504, closed="left", min_periods=0).sum().reset_index()
            # month_rolled_pitcher_df = pitcher_df[['pitcher'] + [col for col in pitcher_df if "month_" in col]].copy().groupby(by="pitcher").rolling(window=75, closed="left", min_periods=0).sum().reset_index()

            # # Assign the rolled values from players' stats back to the player DataFrames 
            # for play in constants.PLAY_TYPES:

            #     batter_df[f"season_{play}"] = season_rolled_batter_df[f"season_{play}"]
            #     batter_df[f"month_{play}"] = month_rolled_batter_df[f"month_{play}"]
                
            #     pitcher_df[f"season_{play}"] = season_rolled_pitcher_df[f"season_{play}"]
            #     pitcher_df[f"month_{play}"] = month_rolled_pitcher_df[f"month_{play}"]

        ######################### REPERCENTAGE NEUTRALIZED STATS (TO SUM % TO 1.0) #########################

            # # Repercentage factored batting stats percentage to sum to 1 because they don't necessarily after neutralization
            # season_columns = ["season_{}".format(play) for play in constants.PLAY_TYPES]
            # month_columns = ["month_{}".format(play) for play in constants.PLAY_TYPES]
            
            # batter_df[season_columns] = batter_df[season_columns].div(batter_df[season_columns].sum(axis=1), axis=0)
            # batter_df[month_columns] = batter_df[month_columns].div(batter_df[month_columns].sum(axis=1), axis=0)

            # # batter_df[season_columns] = batter_df.apply(lambda row: pd.Series([row[f"season_{play_type}"]/row[season_columns].sum() for play_type in list(constants.PLAY_TYPES)]) if row[season_columns].sum() > 0 else pd.Series([0 for play_type in constants.PLAY_TYPES]), axis=1)
            # # batter_df[month_columns] = batter_df.apply(lambda row: pd.Series([row[f"month_{play_type}"]/row[month_columns].sum() for play_type in list(constants.PLAY_TYPES)]) if row[month_columns].sum() > 0 else pd.Series([0 for play_type in constants.PLAY_TYPES]), axis=1)

            # # Repercentage factored pitching stats percentage to sum to 1 because they don't necessarily after neutralization
            
            # pitcher_df[season_columns] = pitcher_df[season_columns].div(pitcher_df[season_columns].sum())
            # pitcher_df[month_columns] = pitcher_df[month_columns].div(pitcher_df[month_columns].sum())
            # # pitcher_df[season_columns] = pitcher_df.apply(lambda row: pd.Series([row[f"season_{play_type}"]/row[season_columns].sum() for play_type in list(constants.PLAY_TYPES)]) if row[season_columns].sum() > 0 else pd.Series([0 for play_type in constants.PLAY_TYPES]), axis=1)
            # # pitcher_df[month_columns] = pitcher_df.apply(lambda row: pd.Series([row[f"month_{play_type}"]/row[month_columns].sum() for play_type in list(constants.PLAY_TYPES)]) if row[month_columns].sum() > 0 else pd.Series([0 for play_type in constants.PLAY_TYPES]), axis=1)

    ######################### STORE FINAL DATAFRAMES #########################

            # Place the final rolling factored batting stats DataFrame into the storage dictionary
            rolling_factored_batting_stats[pitbat_combo] = batter_df[["play_type", "game_pk", "game_date", "ballpark","temprature", "wind_speed", "wind_direction", "batter", "pitcher", "pitbat",'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot', "bat_score", "fld_score"] + [f"{rolling_window_length}_PA_{play}" for rolling_window_length in rolling_windows for play in constants.PLAY_TYPES]]
            rolling_factored_pitching_stats[pitbat_combo] = pitcher_df[["play_type", "game_pk", "game_date", "ballpark","temprature", "wind_speed", "wind_direction", "batter", "pitcher", "pitbat",'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot', "bat_score", "fld_score"] + [f"{rolling_window_length}_PA_{play}" for rolling_window_length in rolling_windows for play in constants.PLAY_TYPES]]
            
        clear_output(wait=False)
        
        return {"pitching_stats":rolling_factored_pitching_stats, "batting_stats":rolling_factored_batting_stats}


    def stitch_pitbat_stats(self, rolling_factored_stats):
        """
        Combines batting and pitching statistics for each batter-pitcher handedness combination (pitbat combo) 
        into a single DataFrame for both batting and pitching stats. This function is used after the stats have 
        been rolled and neutralized, and it concatenates the data for each `pitbat_combo` into one unified 
        DataFrame for each category.

        Parameters:
        --------------
        rolling_factored_stats (dict): 
            A dictionary containing the rolled and neutralized batting and pitching stats, with keys "batting_stats" 
            and "pitching_stats". Each of these contains a dictionary where each key corresponds to a 
            `pitbat_combo` (e.g., left-handed batter vs right-handed pitcher), and each value is a DataFrame with 
            the individual player's play data and their rolled statistics.

        -----------------
        Returns:
        --------------
        dict: A dictionary containing two DataFrames:
            - "batting_stats": A DataFrame with concatenated batting stats for all `pitbat_combo` combinations.
            - "pitching_stats": A DataFrame with concatenated pitching stats for all `pitbat_combo` combinations.

        The returned dictionary structure is as follows:
        {
            "batting_stats": DataFrame with concatenated batting stats,
            "pitching_stats": DataFrame with concatenated pitching stats
        }

        Example:
        - If the input `rolling_factored_stats` contains multiple DataFrames for different handedness combinations, 
        this function will combine them into a single DataFrame for batting and pitching stats.
        """
        # Create Storage
        stitched_data = {}
        
        # Concat all 4 DataFrames (from each pitbat combo) into one dataframe
        df_batter = pd.concat([rolling_factored_stats["batting_stats"][pitbat_combo] for pitbat_combo in constants.HAND_COMBOS])
        df_pitcher = pd.concat([rolling_factored_stats["pitching_stats"][pitbat_combo] for pitbat_combo in constants.HAND_COMBOS])
                
        stitched_data["batting_stats"] = df_batter
        stitched_data["pitching_stats"] = df_pitcher
        
        return stitched_data


    # Attach the pitching probability vector to the training set by "joining" on the pitbat combo, year, and pitcher name, where the date is just less than the given PA.
    # Then reattatch the weather and ballpark info for that game
    def merge_pitching_batting_leagueaverage_and_weather_datasets(self, stitched_dataset, cleaned_raw_pitches):
        """
        Merges batting, pitching, league averages, and weather data into a final dataset for analysis.

        This function performs several merges to combine:
        - Batting statistics with pitching statistics.
        - Batting statistics with weather data based on the game date.
        - Batting statistics with league averages for each play type for the last season and month.

        The function performs the following steps:
        1. Adds pitching statistics to the batting dataset based on the pitbat combo (combination of batter/pitcher handedness).
        2. Merges weather data into the batting dataset, ensuring that weather information is associated with the correct game date.
        3. Calculates league averages for each play type in the past season and month based on the game date.
        4. Cleans up unnecessary columns and adds a binary `is_on_base` column to identify if the batter is on base.

        Parameters:
        -----------
        stitched_dataset : dict
            A dictionary containing the following DataFrames:
            - "batting_stats" (DataFrame): Contains the batting statistics for each game.
            - "pitching_stats" (DataFrame): Contains the pitching statistics for each game.
            
        cleaned_raw_pitches : dict
            A dictionary where keys are `pitbat` combinations (e.g., 'R/L', 'L/R') and values are DataFrames containing weather data for each game.
        
        Returns:
        --------
        DataFrame
            The updated "batting_stats" DataFrame with merged pitching statistics, weather data, league averages, and the new `is_on_base` column.
        
        Notes:
        ------
        - The weather data is merged based on the `game_pk` (game identifier).
        - League averages are calculated using rolling windows for the past 365 days (season) and the past 30 days (month).
        - Missing weather data will be filled with NaN values.
        - The `is_on_base` column is binary: 1 if the player is on base (e.g., single, double, triple, home run, walk), otherwise 0.
        """

        ########################## MERGE BATTING AND PITCHING ##########################
        # Label all the columns as pitcher related in the df for when they are merged later. Then define the set of columns we will need to merge with the total batting stats
        stitched_dataset["pitching_stats"].columns = ["pitcher_" + col for col in stitched_dataset["pitching_stats"].columns]
        pitching_columns_to_add = [f"pitcher_{rolling_window_length}_PA_{play}" for rolling_window_length in self.rolling_windows for play in constants.PLAY_TYPES]

        # Attatch the pitching stats to the batting stats. This works in concept because the indexes remain the same even as the DFs are separated
        stitched_dataset["batting_stats"][pitching_columns_to_add] = stitched_dataset["pitching_stats"][pitching_columns_to_add]

        ########################## MERGE WITH WEATHER ##########################

        # Attatch the weather information # THIS MAY HAVE TO CHANGE WITH WEATHER CODING UPDATES
        if self.verbose:
            print("Attatching Original Weather Information to Final Dataset")

        # Step 1: Flatten the cleaned_raw_pitches dictionary into a single DataFrame
        weather_data_list = []
        for pitbat, df in cleaned_raw_pitches.items():
            df_copy = df.copy()
            df_copy['pitbat'] = pitbat  # Add the pitbat identifier to the DataFrame
            weather_data_list.append(df_copy)

        # Combine all the DataFrames into one
        all_weather_data = pd.concat(weather_data_list)

        # Select only necessary columns (game_pk and weather_columns)
        weather_columns = ["temprature", "Left to Right", "Right to Left", "in", "out", "zero"]
        all_weather_data = all_weather_data[['pitbat', 'game_pk'] + weather_columns].drop_duplicates(subset=['game_pk'])
        # Step 2: Merge the weather data back into the main dataset
        stitched_dataset["batting_stats"] = pd.merge(
            stitched_dataset["batting_stats"],
            all_weather_data,
            on=['game_pk'],  # Merge on both pitbat and game_pk
            how='left'  # Use left join to preserve rows from batting_stats
        )

        # Rename the play_type_x column so that it can be referenced later on.
        stitched_dataset['batting_stats'].rename(columns = {'play_type_x':'play_type', 'pitbat_x':'pitbat'}, inplace=True)

        # Now, any missing weather data will automatically be filled with NaN (which you can convert to None if needed)
        weather_columns.remove('temprature')
        weather_columns.extend(['temprature_x', 'temprature_y'])
        #stitched_dataset["batting_stats"][weather_columns] = stitched_dataset["batting_stats"][weather_columns].fillna(value = None)

        # Convert temprature to temprature squared and drop regular temprature from the DataFrame
        stitched_dataset['batting_stats']["temprature_sq"] = stitched_dataset['batting_stats'].temprature_x.apply(lambda x: x**2)
        stitched_dataset['batting_stats'] = stitched_dataset['batting_stats'].drop(columns=['temprature_x', 'temprature_y'])
        ########################## MERGE WITH LEAGUE AVERAGE INFO ##########################

        # Convert game_date to datetime if it's not already
        stitched_dataset["batting_stats"]["game_date"] = pd.to_datetime(stitched_dataset["batting_stats"]["game_date"])

        # Precompute last month and last season league averages for each pitbat combo
        league_averages_list = []

        # Iterate over each pitbat combo
        for pitbat_combo in constants.HAND_COMBOS:
            pitbat_df = stitched_dataset["batting_stats"][stitched_dataset["batting_stats"].pitbat == pitbat_combo].copy()
            
            # Iterate over unique dates
            for date in pitbat_df.game_date.unique():
                back_days = [date - timedelta(days = rolling_window_length / 2.25) for rolling_window_length in self.rolling_windows]
                # season_ago = date - timedelta(days=365)  # 1 year ago
                # month_ago = date - timedelta(days=30)    # 1 month ago
                
                date_filtered_league_stats = {}

                # Make a df for each of the time frames
                for back_day in back_days:
                    df = pitbat_df[(pitbat_df.game_date < date) & (pitbat_df.game_date > back_day)]
                    date_filtered_league_stats[self.rolling_windows[back_days.index(back_day)]] = df

                # # Filter data within the last month and season
                # season_pitbat_date_df = pitbat_df[(pitbat_df.game_date < date) & (pitbat_df.game_date > season_ago)]
                # month_pitbat_date_df = pitbat_df[(pitbat_df.game_date < date) & (pitbat_df.game_date > month_ago)]
                
                
                # Precompute league averages for each play type and store them in a list
                play_averages = {}
                for play in constants.PLAY_TYPES:
                    play_averages[play] = {}
                    for rolling_window_length in self.rolling_windows:
                        play_average = len(date_filtered_league_stats[rolling_window_length][date_filtered_league_stats[rolling_window_length].play_type == play]) / len(date_filtered_league_stats[rolling_window_length]) if len(date_filtered_league_stats[rolling_window_length]) > 0 else None
                        play_averages[play][rolling_window_length] = play_average
                    
                    league_averages_list.append({
                        "pitbat": pitbat_combo,
                        "game_date": date,
                        "play_type": play} |{f"LA_{rolling_window_length}_PA_{play}":play_averages[play][rolling_window_length] for rolling_window_length in self.rolling_windows})

        # Create a DataFrame from the list of league averages, by combining all they play types from the league averages list for each gamedate/pitbat combo
        merged_data = defaultdict(dict)

        for row in league_averages_list:
            key = (row['pitbat'], row['game_date'])  # Unique identifier
            play_type = row['play_type']            # Identify the play type
            for k, v in row.items():
                if k not in ['pitbat', 'game_date', 'play_type']:  # Exclude grouping keys
                    # Combine play type into the key name for merged data
                    merged_data[key][f"{play_type}_{k}"] = v

        # Step 2: Convert merged data back to a list of dictionaries
        result = [
            {'pitbat': key[0], 'game_date': key[1], **values}
            for key, values in merged_data.items()
        ]

        # Convert to DataFrame
        league_averages_df = pd.DataFrame(result)

        # Merge precomputed league averages back into the main DataFrame
        stitched_dataset["batting_stats"] = pd.merge(
            stitched_dataset["batting_stats"],
            league_averages_df,
            on=["pitbat", "game_date"],
            how="left"
        )
        ########################## ADD FINAL TOUCHES ##########################
        
        # Remove unwanted columns
        stitched_dataset['batting_stats'] = stitched_dataset['batting_stats'][[col for col in stitched_dataset['batting_stats'].columns if col not in ["game_pk", "batter", "pitcher", "wind_speed", "wind_direction", "year", 'pitbat_y']]]
        
        # Finally, add a column that is a binary 'is on base' in case we want to run a two step prediction algorithm with step one on base and step two what kind of on base or out
        stitched_dataset["batting_stats"]["is_on_base"] = stitched_dataset["batting_stats"].play_type.apply(lambda x: 1 if x in ["single", "double", "triple", "home_run", "walk", "intent_walk"] else 0)

        clear_output(wait=False)
        
        return stitched_dataset['batting_stats']


    def make_dataset_machine_trainable(self, final_dataset):
        """
        Prepares the given dataset for machine learning by performing the following transformations:
        1. Converts binary-type text columns into actual binary.
        2. Drops rows with missing values and filters out games before May for training.
        3. Removes the 'game_date' column, which is no longer needed.
        4. Defines the target variables (y_play and y_onbase) and drops them from the dataset.
        5. Scales numeric features and encodes categorical features (using one-hot encoding).
        6. Returns the preprocessed dataset along with the target variables.

        Args:
            final_dataset (pd.DataFrame): The dataset to be prepared for machine learning, which should
                                        include columns such as 'on_3b', 'on_2b', 'on_1b', 'inning_topbot',
                                        'game_date', 'play_type', and 'is_on_base'.
        
        Returns:
            dict: A dictionary containing:
                - "X" (np.ndarray): The preprocessed features (numeric and encoded categorical features).
                - "y_play" (pd.Series): The target variable for the type of play ('play_type').
                - "y_onbase" (pd.Series): The target variable indicating if the player is on base ('is_on_base').

        Example:
            prepared_data = make_dataset_machine_trainable(dataset)
            X = prepared_data["X"]
            y_play = prepared_data["y_play"]
            y_onbase = prepared_data["y_onbase"]
        """
        
        # Convert some binary type text columns into actual binary
        for col in ["on_3b", "on_2b", "on_1b"]:
            final_dataset[col] = final_dataset[col].apply(lambda x: 1 if pd.isna(x) == False else 0) 
        final_dataset["inning_topbot"] = final_dataset[col].apply(lambda x: 1 if x == "Top" else 0) 
        
        # Drop NA rows and games before May for training purposes
        final_dataset = final_dataset.dropna()

        final_dataset = final_dataset[final_dataset.game_date.apply(lambda x: x.month >= 5)].reset_index(drop=True)

        # Drop the game date column. We couldn't do this earlier because we needed it to filter out early season games in the line before
        final_dataset.drop(columns = ["game_date"], inplace=True)

        # Define our y targets and drop them from the dataset
        y_play = final_dataset.play_type
        y_onbase = final_dataset.is_on_base

        # And drop the columns
        final_dataset.drop(columns = ["play_type", "is_on_base"], inplace = True)

        # Create a pipeline for scaling, and encoding (and eventually PCA if needed)
        numeric_features = [col for col in final_dataset if col not in ["ballpark", "pitbat"]]

        numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]
        )
        
        categorical_features = ["ballpark", "pitbat"]
        categorical_transformer = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        pipe = Pipeline(
        steps=[("preprocessor", preprocessor)]
        )

        final_dataset = pipe.fit_transform(final_dataset)

        return {"X":final_dataset, "y_play":y_play, "y_onbase":y_onbase}


    def calculate_league_averages(self, neutralized_unrolled_data): # The input here is the output of the neutralize_stats function
        '''
        This is used to calculate the league averages over a period of time. To be used in creating a baseline guesser when used over the entire dataset
        '''
        league_average_plays_dict = {}
        for pitbat_combo in constants.HAND_COMBOS:
            league_average_plays_dict[pitbat_combo] = {}
            for play in constants.PLAY_TYPES:
                df = neutralized_unrolled_data[pitbat_combo]
                play_share = len(df[df.play_type == play])/len(df)
                league_average_plays_dict[pitbat_combo][play] = play_share

        return league_average_plays_dict


    ############################### CREATING FINAL DATASETS  ###############################

    def _make_final_dataset(self, cleaned_pitches, coef_dicts):
        """
        Prepares the final dataset by neutralizing, rolling, and merging pitching, batting, league averages,
        and weather data.

        Args:
            cleaned_pitches (dict): Cleaned pitch data for each 'pitbat' combo.
            coef_dicts (dict): Coefficients for neutralizing stats.

        Returns:
            pd.DataFrame: Final processed dataset with neutralized stats, rolling averages, and merged data.

        Example:
            final_dataset = _make_final_dataset(cleaned_pitches, coef_dicts)
        """

        neutralized_data = self.neutralize_stats(cleaned_pitches, coef_dicts)

        rolled_stats = self.roll_neutralized_batting_stats(neutralized_data)

        stitched_stats = self.stitch_pitbat_stats(rolled_stats)

        final_dataset = self.merge_pitching_batting_leagueaverage_and_weather_datasets(stitched_stats, cleaned_pitches)
        
        return final_dataset

    def build_training_dataset(self, raw_pitches, suffix, save_cleaned=False, save_coefficients=False,
                            save_dataset=False, save_training_dataset=False, online_save = False, local_save=False):
        """
        Cleans raw pitch data, generates neutralization coefficients, builds a final dataset, and prepares
        a machine-readable training dataset. Optionally saves all intermediate results.

        Args:
            raw_pitches (dict): Raw pitch data for each 'pitbat' combo.
            suffix (str): Suffix for file names.
            save_cleaned (bool): Whether to save cleaned data.
            save_coefficients (bool): Whether to save neutralization coefficients.
            save_dataset (bool): Whether to save the final dataset.
            save_training_dataset (bool): Whether to save the training dataset.
            online_save (bool): Whether to save data to the cloud.
            local_save (bool): Whether to save data locally.

        Returns:
            dict: Training dataset dictionary containing features and target values.
            
        Example:
            training_data = build_training_dataset(raw_pitches, 'model_v1', save_cleaned=True, online_save=True)
        """

        # Clean raw pitches and return a cleaned pitches DataFrame
        cleaned_data = self.clean_raw_pitches(raw_pitches)

        if save_cleaned:
            if online_save: 
                # Convert the dict of dataframes to json so it can be uploaded
                cleaned_data_json = {df_name: df.to_json() for df_name, df in cleaned_data.items()}
                cf.CloudHelper(obj=cleaned_data_json).upload_to_cloud('simulation_training_data', f"cleaned_data_{suffix}")
            if local_save:
                with open(f"Data/cleaned_data_{suffix}", 'wb') as f:
                    pkl.dump(cleaned_data, f)

        # Create a neutralization coefficients dictionary
        coef_dicts = self.build_neutralization_coefficient_dictionaries(cleaned_data)
        if save_coefficients: 
            if online_save:
                cf.CloudHelper(obj=coef_dicts).upload_to_cloud('simulation_training_data', f"neutralization_coefficients_dict_{suffix}")
            if local_save:
                with open(f"Data/neutralization_coefficients_dict_{suffix}", 'wb') as f:
                    pkl.dump(coef_dicts, f)


        # Build the final dataset, and machine readable training set
        final_dataset = self._make_final_dataset(cleaned_data, coef_dicts)
        if save_dataset:
            if online_save:
                cf.CloudHelper(obj=final_dataset).upload_to_cloud('simulation_training_data', f"Final Datasets/final_dataset_{suffix}")
            if local_save:
                with open(f"Data/final_dataset_nonML_{suffix}", 'wb') as f:
                    pkl.dump(final_dataset, f)


        training_dataset = self.make_dataset_machine_trainable(final_dataset)
        if save_training_dataset:
            if online_save:
                # First convert the dict of nd.arrays to json for uploading
                list_dataset = {name: array.tolist() for name, array in training_dataset.items()}
                training_dataset_json = {array_name: json.dumps(array) for array_name, array in list_dataset.items()}
                cf.CloudHelper(obj=training_dataset_json).upload_to_cloud('simulation_training_data', f"Training Datasets/training_dataset{suffix}")
            if local_save:
                with open(f"Data/final_dataset_ML_{suffix}", 'wb') as f:
                    pkl.dump(training_dataset, f)


