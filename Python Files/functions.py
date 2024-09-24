import constants

import pandas as pd
import numpy as np
import sys
from IPython.display import clear_output

# Change the path so that we can import the local cloud functions stored in a different directory. THE PATH IS DIFFERENT ON MAC AND PI, SO USE TRY EXCEPT FOR BOTH
try: # for mac
    sys.path.insert(1, '/users/jaredzirkes/Desktop/Python/GitHub')
    from google_cloud.cloud_functions import CloudHelper
except:
    sys.path.insert(1, "/home/pi/Desktop/Python")
    from google_cloud.cloud_functions import CloudHelper

######################################################################################
# FUNCTIONS TO BE USED IN BUILDING A TRAINING SET FOR PA PROBABILITY PREDICTIONS
######################################################################################


######################################################################################
# Clean Pitch Data
######################################################################################
def get_wind_direction(full_weather: str) -> str:

    """Given the full weather description as scraped from a baseball reference box score like
    https://www.baseball-reference.com/boxes/CHA/CHA202407080.shtml, pull out just the wind direction.
    
    ------------INPUTS------------
    full_weather: String
        - The raw string of weather information as pulled from a baseball reference box score

    ------------OUTPUTS------------
    weather: String
        - A shortened string of the weather, one of ["in", "out", "Right to Left", "Left to Right"]

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

def convert_wind_direction(all_plays_by_pitbat_combo, wind_column = "wind_direction"):
    """
    Function converts the wind columns in all_plays_by_hand_combo from a categorical wind direction (string) and numeric wind speed into OHE columns representing
    both wind direction and wind speed
    
    Parameters
    --------------
    all_plays_by_hand_combo: DataFrame
        A cleaned DataFrame of pitches, including columns for the wind direction and wind speed of each play
    -----------------    
   
    Returns: Dataframe
        A DataFrame of all pitches divided by pitbat combo, now including a set of columns, one each for each possible wind direction, with values of the wind
        speed in that direction
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

def correct_home_away_swap(all_pitches: pd.DataFrame) -> pd.DataFrame:
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

def segregate_plays_by_pitbat_combo(cleaned_plays: pd.DataFrame) -> pd.DataFrame:
    """Function that divides a full dataframe of cleaned plays into a dictionary, segregating the plays by the 4 possible pitbat combos.
    
    ------------INPUTS------------
    cleaned_plays: DataFrame
        - A dataframe of cleaned plays, can be the output of the clean_raw_pitches function
        
    ------------OUTPUTS------------
    all_plays_by_pitbat_combo: Dictionary
        - A dictionary with 4 keys, 1 for each of the possible pitbat combos. Within each entry is a dataframe of all the plays from the
        input dataframe belonging to the given pitbat combo.

    """
    # Set up a Dictionary to hold all pitches, divided by the pitbat combo
    all_plays_by_pitbat_combo = {"RR":{}, "RL":{}, "LR":{}, "LL":{}}

    # Split all plays on combo of batter/pitcher handedness before placing into the dictionary
    for pitbat_combo in all_plays_by_pitbat_combo.keys(): 
        pitbat_df = cleaned_plays[(cleaned_plays.stand == pitbat_combo[0]) & (cleaned_plays.p_throws == pitbat_combo[1])].copy().reset_index(drop=True)
        all_plays_by_pitbat_combo[pitbat_combo]= pitbat_df

    return all_plays_by_pitbat_combo




################################################################################
def clean_raw_pitches(raw_pitches_df: pd.DataFrame) -> pd.DataFrame:
    """Function to clean a dataframe of raw pitches into a dataframe that we can use for all our later analyses.
   
    ------------INPUTS------------
    raw_pitches_df: DataFrame
        - A dataframe of uncleaned pitches from the statcast API.
        
    ------------OUTPUTS------------
    cleaned_pitches: DataFrame
        - A cleaned dataframe of pitches resulting in plays."""

    # Filter down to only regular season games
    raw_pitches_df = raw_pitches_df[raw_pitches_df.game_type == "R"]

    # Correct home and away mistakes in the pitch data
    raw_pitches_df = correct_home_away_swap(raw_pitches_df)

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

   
    ############ Attatch the weather information to each pitch ############

    # Build a combined weather dataframe by concatanating all yearly weather dataframes belonging to years present in the plays dataframe
    years = list(final_plays.game_date.apply(lambda x: x.split("-")[0]).value_counts().index)
    weather_dictionary_holder = {}

    for year in years:
        yearly_weather_df = CloudHelper().download_from_cloud("proreference_weather_data/weather_data_{}".format(year))
        weather_dictionary_holder[year] = yearly_weather_df
    total_weather_df = pd.concat([df for df in weather_dictionary_holder.values()])
    
    # Attatch the raw weather string to the the play by matching the date and home team
    final_plays["full_weather"] = final_plays.apply(lambda x: total_weather_df[(total_weather_df.date.values == x.game_date) & ((total_weather_df.home_team.values == constants.WEATHER_NAME_CONVERSIONS[x.home_team])|((total_weather_df.home_team.values == constants.WEATHER_NAME_CONVERSIONS[x.away_team])))].weather.iloc[0], axis = 1)                                                                                        
    # Break up the full weather info into temp, wind speed, and wind direction seperately
    final_plays["temprature"] = final_plays.full_weather.apply(lambda x: int(x.split(": ")[1].split("°")[0]))
    final_plays["wind_speed"] = final_plays.full_weather.apply(lambda x: int(x.split("Wind ")[1].split("mph")[0]) if "Wind" in x else 0)
    final_plays["wind_direction"] = final_plays.full_weather.apply(get_wind_direction)
    final_plays["wind_direction"] = final_plays.wind_direction.apply(lambda x: x.split(", ")[0] if x != None else x)

    # Convert the wind direction text column into a one-hot encoded set of columns multiplied by the wind speed (yields individual columns representing total wind speed)
    final_plays = convert_wind_direction(final_plays, final_plays.wind_direction)


    ############ Attatch the ballpark info to each pitch ############

    # Import file to help connect team and year with a specific ballpark
    ballpark_info = pd.read_excel("../Outside Info Files/Ballpark Info.xlsx", header=2)[["Stadium", "Team", "Start Date", "End Date"]]
        
    # Create a column for the ballpark based on the date and home_team of each pitch
    final_plays["ballpark"] = final_plays.apply(lambda x: ballpark_info[(ballpark_info.Team.values == x.home_team) & (ballpark_info["End Date"].values > int(x.game_date.split("-")[0]))].Stadium.iloc[0],axis=1)
    
    ############ Divide pitches by pitbat combos in 4 dataframes ############
    all_plays_by_pitbat_combo = segregate_plays_by_pitbat_combo(final_plays)

    return all_plays_by_pitbat_combo

#####################################################################################

######################################################################################
# Build a DataFrame for Weather Regression
######################################################################################

def insert_game_play_shares(all_plays_by_pitbat_combo: dict) -> dict:
    """
    Function calculates the game play share (percentage outcome by play type) for individual plays for a specific game and inserts a
    column of the shares into all relevant all plays dfs
    
    Parameters
    --------------
    all_plays_by_hand_combo: DataFrame
        A cleaned DataFrame of pitches (will be divided by pitbat combo), including columns for the play type of each play and the game_pk for each game
    -----------------    
   
    Returns: Dataframe
        A DataFrame of all pitches, now including a column for the game play share of each play type within each different game
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
            if n%1000 == 0:
                print("Calculating The Play Share by Play Type for Each Game. There are {}K Instances Remaining".format(round((sum([len(all_plays_by_pitbat_combo[x].game_pk.unique()) for x in constants.HAND_COMBOS])-n)/1000),6))
            n+= 1
        clear_output(wait = True)
    

    # Add a column in the all plays dfs that is the game play share for the specific game and play type of each play
    for pitbat_combo in constants.HAND_COMBOS:
        print("Inserting Play Shares by Play Type from Each Game To the All Pitches Data Set. There are {} Pitbat Combos Remaining".format(len(constants.HAND_COMBOS) - constants.HAND_COMBOS.index(pitbat_combo)))
        clear_output(wait = True)
        
        # The if statement in the apply below is used to catch the rare case (n=2 PA in 2018-2019) where the game_pk = <NA>. When this happens the play associated is in the game itself, but does not make it into the 
        # game_play_shares dict which throws and error when pulling the play type from the dictionary
        game_play_df = all_plays_by_pitbat_combo[pitbat_combo].copy()
        game_play_df["game_play_share"] = game_play_df.apply(lambda x: game_play_shares[pitbat_combo][x.game_pk].loc[x.play_type].play_share if x.play_type in game_play_shares[pitbat_combo][x.game_pk].index else 0, axis = 1)
            
    return all_plays_by_pitbat_combo

def insert_missing_game_play_shares(weather_regression_data: dict, hand_combos: list = constants.HAND_COMBOS) -> dict:
    # As the only plays in our data are types that happened in games, fill in all the missing play types for each game with a game_share of 0 for that play type
    play_types = constants.PLAY_TYPES
    n = 0
    for pitbat_combo in constants.HAND_COMBOS:
        for game in weather_regression_data[pitbat_combo].game_pk.unique():
            n += 1
            if n%500 == 0:
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
                                                                                                                         "wind_speed":[game_info.wind_speed]*num_missing_plays,
                                                                                                                         "wind_direction":[game_info.wind_direction]*num_missing_plays,
                                                                                                                         "game_play_share":[0]*num_missing_plays})])

    return weather_regression_data

###########################################################################################    
def create_weather_regression_dataframes(all_plays_by_hand_combo):
    """ INSERT FUNCTION INFORMATION"""

    # Start by filling in all the game play shares
    games_df = insert_game_play_shares(all_plays_by_hand_combo.copy())
    games_df = insert_missing_game_play_shares(games_df)

    weather_training_data = {x:{} for x in constants.HAND_COMBOS}
    l =  []

    # Clean the data to fit what we will need for weather regressions
    for pitbat_combo in constants.HAND_COMBOS:  
        weather_training_df = games_df[pitbat_combo].copy()
        
        # Remove any games with a month lower than 5 (May)
        weather_training_df = weather_training_df[weather_training_df.game_date.apply(lambda x: int(x.split("-")[1])) >=5]
        
        # Filter to only the columns we will need for the weather regressions
        weather_training_data[pitbat_combo] = weather_training_df[["game_pk", "play_type", "temprature", "wind_speed", "wind_direction", "game_play_share"]]
        
        # Square temprature to use in the regression because I believe it behaves this way
        weather_training_data[pitbat_combo]["temprature_squared"] = weather_training_data[pitbat_combo]["temprature"].apply(lambda x: x**2)

        # Group the weather training data by game and play type to get the game_play_share for each play type for each game
        weather_training_data[pitbat_combo] = weather_training_data[pitbat_combo].groupby(by = ["game_pk", "play_type"]).last().reset_index()
    
    clear_output(wait = False)
        
        
    return weather_training_data
###########################################################################################

