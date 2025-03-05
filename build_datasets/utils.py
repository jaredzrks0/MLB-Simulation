import pandas as pd

def _get_wind_direction(full_weather: str) -> str:
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
            # full_weather.full_weather.split("mph ")[-1].split(' from')[0]
            weather = "in"
        elif "out" in "".join(full_weather.split("Wind")) or "Out" in "".join(full_weather.split("Wind")):
            # full_weather.full_weather.split("mph ")[-1].split(' to')[0]
            weather = "out"
        elif "Left" in "".join(full_weather.split("Wind")) or "Right" in "".join(full_weather.split("Wind")):
            weather = full_weather.split(
                "from ")[-1].strip(".").split(", ")[0]
        else:  # Sometimes wind just listed as 0mph with no direction
            weather = None
    else:
        weather = None

    return weather

def _convert_wind_direction(all_plays_by_pitbat_combo, wind_column="wind_direction"):
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
    wind_columns = pd.get_dummies(
        wind_column, columns=['categorical_column', ])
    wind_columns = pd.concat(
        [all_plays_by_pitbat_combo, wind_columns], axis=1)

    # Finally multiply the binary wind direction columns by the wind speed to get the final wind speed in the correct direction
    for column in wind_columns.columns[-5:]:
        wind_columns[column] = wind_columns[column] * \
            wind_columns["wind_speed"]

    return wind_columns

def _pull_full_weather(game_date, home_team, away_team, total_weather_df):
    try:
        # value = total_weather_df[(total_weather_df.date.values == game_date) & ((total_weather_df.home_team.values == constants.WEATHER_NAME_CONVERSIONS[home_team])|(total_weather_df.home_team.values == constants.WEATHER_NAME_CONVERSIONS[away_team]))].weather.iloc[0]
        value = total_weather_df[(total_weather_df.date.values == game_date) & ((total_weather_df.converted_home_team.values == home_team) | (
            total_weather_df.converted_home_team.values == away_team))].weather.iloc[0]
        return value
    except:
        return 'Start Time Weather: 72° F, Wind 0mph, In Dome.'

def _segregate_plays_by_pitbat_combo(cleaned_plays: pd.DataFrame) -> pd.DataFrame:
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
    all_plays_by_pitbat_combo = {"RR": {}, "RL": {}, "LR": {}, "LL": {}}

    # Split all plays on combo of batter/pitcher handedness before placing into the dictionary
    for pitbat_combo in all_plays_by_pitbat_combo.keys():
        pitbat_df = cleaned_plays[(cleaned_plays.stand == pitbat_combo[0]) & (
            cleaned_plays.p_throws == pitbat_combo[1])].copy().reset_index(drop=True)
        all_plays_by_pitbat_combo[pitbat_combo] = pitbat_df

    return all_plays_by_pitbat_combo

def _correct_home_away_swap(all_pitches: pd.DataFrame) -> pd.DataFrame:
        """Function used to correct a series of games across years where the home team and away team are swapped on baseball reference

        ------------INPUTS------------
        - all_pitches: DataFrame
            A dataframe of individual pitches, pulled from the statcast API.

        ------------OUTPUTS------------
        - all_pitches: DataFrame
            A dataframe of individual pitches, identical to the function's input, other than the correction of home and away teams in
            a select subset of games.

        """
        strange_games_I = all_pitches[(all_pitches.home_team == "TOR") & (
            all_pitches.away_team == "WSH")].index
        all_pitches.loc[strange_games_I, "home_team"] = "WSH"
        all_pitches.loc[strange_games_I, "away_team"] = "TOR"

        strange_games_II = all_pitches[(all_pitches.home_team == "CIN") & (
            all_pitches.away_team == "SF") & (all_pitches.game_date == "2013-07-23")].index
        all_pitches.loc[strange_games_II, "home_team"] = "SF"
        all_pitches.loc[strange_games_II, "away_team"] = "CIN"

        strange_games_III = all_pitches[(all_pitches.home_team == "BAL") & (
            all_pitches.away_team == "TB") & (all_pitches.game_date == "2015-05-01")].index
        all_pitches.loc[strange_games_III, "home_team"] = "TB"
        all_pitches.loc[strange_games_III, "away_team"] = "BAL"

        strange_games_IV = all_pitches[(all_pitches.home_team == "BAL") & (
            all_pitches.away_team == "TB") & (all_pitches.game_date == "2015-05-02")].index
        all_pitches.loc[strange_games_IV, "home_team"] = "TB"
        all_pitches.loc[strange_games_IV, "away_team"] = "BAL"

        strange_games_V = all_pitches[(all_pitches.home_team == "BAL") & (
            all_pitches.away_team == "TB") & (all_pitches.game_date == "2015-05-03")].index
        all_pitches.loc[strange_games_V, "home_team"] = "TB"
        all_pitches.loc[strange_games_V, "away_team"] = "BAL"

        strange_games_VI = all_pitches[(all_pitches.home_team == "MIA") & (
            all_pitches.away_team == "MIL") & (all_pitches.game_date == "2017-09-16")].index
        all_pitches.loc[strange_games_VI, "home_team"] = "MIL"
        all_pitches.loc[strange_games_VI, "away_team"] = "MIA"

        strange_games_VII = all_pitches[(all_pitches.home_team == "MIA") & (
            all_pitches.away_team == "MIL") & (all_pitches.game_date == "2017-09-17")].index
        all_pitches.loc[strange_games_VII, "home_team"] = "MIL"
        all_pitches.loc[strange_games_VII, "away_team"] = "MIA"

        strange_games_VIII = all_pitches[(all_pitches.home_team == "MIA") & (
            all_pitches.away_team == "MIL") & (all_pitches.game_date == "2017-09-15")].index
        all_pitches.loc[strange_games_VIII, "home_team"] = "MIL"
        all_pitches.loc[strange_games_VIII, "away_team"] = "MIA"

        strange_games_IX = all_pitches[(all_pitches.home_team == "NYY") & (
            all_pitches.away_team == "PHI") & (all_pitches.game_date == "2020-08-05")].index
        all_pitches.loc[strange_games_IX, "home_team"] = "PHI"
        all_pitches.loc[strange_games_IX, "away_team"] = "NYY"

        strange_games_X = all_pitches[(all_pitches.home_team == "MIA") & (
            all_pitches.away_team == "BAL") & (all_pitches.game_date == "2020-08-05")].index
        all_pitches.loc[strange_games_X, "home_team"] = "BAL"
        all_pitches.loc[strange_games_X, "away_team"] = "MIA"

        strange_games_XI = all_pitches[(all_pitches.home_team == "MIA") & (
            all_pitches.away_team == "BAL") & (all_pitches.game_date == "2020-08-06")].index
        all_pitches.loc[strange_games_XI, "home_team"] = "BAL"
        all_pitches.loc[strange_games_XI, "away_team"] = "MIA"

        strange_games_XII = all_pitches[(all_pitches.home_team == "MIA") & (
            all_pitches.away_team == "BAL") & (all_pitches.game_date == "2020-08-07")].index
        all_pitches.loc[strange_games_XII, "home_team"] = "BAL"
        all_pitches.loc[strange_games_XII, "away_team"] = "MIA"

        strange_games_XIII = all_pitches[(all_pitches.home_team == "STL") & (
            all_pitches.away_team == "CHC") & (all_pitches.game_date == "2020-08-17")].index
        all_pitches.loc[strange_games_XIII, "home_team"] = "CHC"
        all_pitches.loc[strange_games_XIII, "away_team"] = "STL"

        strange_games_XIX = all_pitches[(all_pitches.home_team == "STL") & (
            all_pitches.away_team == "CHC") & (all_pitches.game_date == "2020-08-18")].index
        all_pitches.loc[strange_games_XIX, "home_team"] = "CHC"
        all_pitches.loc[strange_games_XIX, "away_team"] = "STL"

        strange_games_XX = all_pitches[(all_pitches.home_team == "STL") & (
            all_pitches.away_team == "CHC") & (all_pitches.game_date == "2020-08-19")].index
        all_pitches.loc[strange_games_XX, "home_team"] = "CHC"
        all_pitches.loc[strange_games_XX, "away_team"] = "STL"

        strange_games_XXI = all_pitches[(all_pitches.home_team == "MIA") & (
            all_pitches.away_team == "WSH") & (all_pitches.game_date == "2020-08-22")].index
        all_pitches.loc[strange_games_XXI, "home_team"] = "WSH"
        all_pitches.loc[strange_games_XXI, "away_team"] = "MIA"

        strange_games_XXII = all_pitches[(all_pitches.home_team == "MIA") & (
            all_pitches.away_team == "NYM") & (all_pitches.game_date == "2020-08-25")].index
        all_pitches.loc[strange_games_XXII, "home_team"] = "NYM"
        all_pitches.loc[strange_games_XXII, "away_team"] = "MIA"

        strange_games_XXIII = all_pitches[(all_pitches.home_team == "NYY") & (
            all_pitches.away_team == "ATL") & (all_pitches.game_date == "2020-08-26")].index
        all_pitches.loc[strange_games_XXIII, "home_team"] = "ATL"
        all_pitches.loc[strange_games_XXIII, "away_team"] = "NYY"

        strange_games_XXIV = all_pitches[(all_pitches.home_team == "CIN") & (
            all_pitches.away_team == "MIL") & (all_pitches.game_date == "2020-08-27")].index
        all_pitches.loc[strange_games_XXIV, "home_team"] = "MIL"
        all_pitches.loc[strange_games_XXIV, "away_team"] = "CIN"

        strange_games_XXV = all_pitches[(all_pitches.home_team == "SEA") & (
            all_pitches.away_team == "SD") & (all_pitches.game_date == "2020-08-27")].index
        all_pitches.loc[strange_games_XXV, "home_team"] = "SD"
        all_pitches.loc[strange_games_XXV, "away_team"] = "SEA"

        strange_games_XXVI = all_pitches[(all_pitches.home_team == "LAD") & (
            all_pitches.away_team == "SF") & (all_pitches.game_date == "2020-08-27")].index
        all_pitches.loc[strange_games_XXVI, "home_team"] = "SF"
        all_pitches.loc[strange_games_XXVI, "away_team"] = "LAD"

        strange_games_XXVII = all_pitches[(all_pitches.home_team == "PIT") & (
            all_pitches.away_team == "STL") & (all_pitches.game_date == "2020-08-27")].index
        all_pitches.loc[strange_games_XXVII, "home_team"] = "STL"
        all_pitches.loc[strange_games_XXVII, "away_team"] = "PIT"

        strange_games_XXVIII = all_pitches[(all_pitches.home_team == "NYM") & (
            all_pitches.away_team == "NYY") & (all_pitches.game_date == "2020-08-28")].index
        all_pitches.loc[strange_games_XXVIII, "home_team"] = "NYY"
        all_pitches.loc[strange_games_XXVIII, "away_team"] = "NYM"

        strange_games_XXVIV = all_pitches[(all_pitches.home_team == "MIN") & (
            all_pitches.away_team == "DET") & (all_pitches.game_date == "2020-08-29")].index
        all_pitches.loc[strange_games_XXVIV, "home_team"] = "DET"
        all_pitches.loc[strange_games_XXVIV, "away_team"] = "MIN"

        strange_games_XXVV = all_pitches[(all_pitches.home_team == "OAK") & (
            all_pitches.away_team == "HOU") & (all_pitches.game_date == "2020-08-29")].index
        all_pitches.loc[strange_games_XXVV, "home_team"] = "HOU"
        all_pitches.loc[strange_games_XXVV, "away_team"] = "OAK"

        strange_games_XXVVI = all_pitches[(all_pitches.home_team == "CHC") & (
            all_pitches.away_team == "CIN") & (all_pitches.game_date == "2020-08-29")].index
        all_pitches.loc[strange_games_XXVVI, "home_team"] = "CIN"
        all_pitches.loc[strange_games_XXVVI, "away_team"] = "CHC"

        strange_games_XXVVII = all_pitches[(all_pitches.home_team == "NYM") & (
            all_pitches.away_team == "NYY") & (all_pitches.game_date == "2020-08-30")].index
        all_pitches.loc[strange_games_XXVVII, "home_team"] = "NYY"
        all_pitches.loc[strange_games_XXVVII, "away_team"] = "NYM"

        strange_games_XXVVIII = all_pitches[(all_pitches.home_team == "WSH") & (
            all_pitches.away_team == "ATL") & (all_pitches.game_date == "2020-09-4")].index
        all_pitches.loc[strange_games_XXVVIII, "home_team"] = "ATL"
        all_pitches.loc[strange_games_XXVVII, "away_team"] = "WSH"

        strange_games_XXVVIV = all_pitches[(all_pitches.home_team == "NYY") & (
            all_pitches.away_team == "BAL") & (all_pitches.game_date == "2020-09-04")].index
        all_pitches.loc[strange_games_XXVVIV, "home_team"] = "BAL"
        all_pitches.loc[strange_games_XXVVIV, "away_team"] = "NYY"

        strange_games_XXVVV = all_pitches[(all_pitches.home_team == "TOR") & (
            all_pitches.away_team == "BOS") & (all_pitches.game_date == "2020-09-04")].index
        all_pitches.loc[strange_games_XXVVV, "home_team"] = "BOS"
        all_pitches.loc[strange_games_XXVVV, "away_team"] = "TOR"

        strange_games_XXVVVI = all_pitches[(all_pitches.home_team == "DET") & (
            all_pitches.away_team == "MIN") & (all_pitches.game_date == "2020-09-04")].index
        all_pitches.loc[strange_games_XXVVVI, "home_team"] = "MIN"
        all_pitches.loc[strange_games_XXVVVI, "away_team"] = "DET"

        strange_games_XXVVVII = all_pitches[(all_pitches.home_team == "CIN") & (
            all_pitches.away_team == "PIT") & (all_pitches.game_date == "2020-09-04")].index
        all_pitches.loc[strange_games_XXVVVII, "home_team"] = "PIT"
        all_pitches.loc[strange_games_XXVVVII, "away_team"] = "CIN"

        strange_games_XXVVVIII = all_pitches[(all_pitches.home_team == "HOU") & (
            all_pitches.away_team == "LAA") & (all_pitches.game_date == "2020-09-05")].index
        all_pitches.loc[strange_games_XXVVVIII, "home_team"] = "LAA"
        all_pitches.loc[strange_games_XXVVVIII, "away_team"] = "HOU"

        strange_games_XXVVVIV = all_pitches[(all_pitches.home_team == "STL") & (
            all_pitches.away_team == "CHC") & (all_pitches.game_date == "2020-09-05")].index
        all_pitches.loc[strange_games_XXVVVIV, "home_team"] = "CHC"
        all_pitches.loc[strange_games_XXVVVIV, "away_team"] = "STL"

        strange_games_L = all_pitches[(all_pitches.home_team == "HOU") & (
            all_pitches.away_team == "OAK") & (all_pitches.game_date == "2020-09-08")].index
        all_pitches.loc[strange_games_L, "home_team"] = "OAK"
        all_pitches.loc[strange_games_L, "away_team"] = "HOU"

        strange_games_LI = all_pitches[(all_pitches.home_team == "BOS") & (
            all_pitches.away_team == "PHI") & (all_pitches.game_date == "2020-09-08")].index
        all_pitches.loc[strange_games_LI, "home_team"] = "PHI"
        all_pitches.loc[strange_games_LI, "away_team"] = "BOS"

        strange_games_LII = all_pitches[(all_pitches.home_team == "MIN") & (
            all_pitches.away_team == "STL") & (all_pitches.game_date == "2020-09-08")].index
        all_pitches.loc[strange_games_LII, "home_team"] = "STL"
        all_pitches.loc[strange_games_LII, "away_team"] = "MIN"

        strange_games_LIII = all_pitches[(all_pitches.home_team == "DET") & (
            all_pitches.away_team == "STL") & (all_pitches.game_date == "2020-09-10")].index
        all_pitches.loc[strange_games_LIII, "home_team"] = "STL"
        all_pitches.loc[strange_games_LIII, "away_team"] = "DET"

        strange_games_LIV = all_pitches[(all_pitches.home_team == "PHI") & (
            all_pitches.away_team == "MIA") & (all_pitches.game_date == "2020-09-11")].index
        all_pitches.loc[strange_games_LIV, "home_team"] = "MIA"
        all_pitches.loc[strange_games_LIV, "away_team"] = "PHI"

        strange_games_LV = all_pitches[(all_pitches.home_team == "BAL") & (
            all_pitches.away_team == "NYY") & (all_pitches.game_date == "2020-09-11")].index
        all_pitches.loc[strange_games_LV, "home_team"] = "NYY"
        all_pitches.loc[strange_games_LV, "away_team"] = "BAL"

        strange_games_LVI = all_pitches[(all_pitches.home_team == "OAK") & (
            all_pitches.away_team == "TEX") & (all_pitches.game_date == "2020-09-12")].index
        all_pitches.loc[strange_games_LVI, "home_team"] = "TEX"
        all_pitches.loc[strange_games_LVI, "away_team"] = "OAK"

        strange_games_LVII = all_pitches[(all_pitches.home_team == "PHI") & (
            all_pitches.away_team == "MIA") & (all_pitches.game_date == "2020-09-13")].index
        all_pitches.loc[strange_games_LVII, "home_team"] = "MIA"
        all_pitches.loc[strange_games_LVII, "away_team"] = "PHI"

        strange_games_LVIII = all_pitches[(all_pitches.home_team == "SF") & (
            all_pitches.away_team == "SD") & (all_pitches.game_date == "2020-09-13")].index
        all_pitches.loc[strange_games_LVII, "home_team"] = "SD"
        all_pitches.loc[strange_games_LVII, "away_team"] = "SF"

        strange_games_LIX = all_pitches[(all_pitches.home_team == "PIT") & (
            all_pitches.away_team == "CIN") & (all_pitches.game_date == "2020-09-14")].index
        all_pitches.loc[strange_games_LIX, "home_team"] = "CIN"
        all_pitches.loc[strange_games_LIX, "away_team"] = "PIT"

        strange_games_LX = all_pitches[(all_pitches.home_team == "STL") & (
            all_pitches.away_team == "MIL") & (all_pitches.game_date == "2020-09-14")].index
        all_pitches.loc[strange_games_LX, "home_team"] = "MIL"
        all_pitches.loc[strange_games_LX, "away_team"] = "STL"

        strange_games_LXI = all_pitches[(all_pitches.home_team == "OAK") & (
            all_pitches.away_team == "SEA") & (all_pitches.game_date == "2020-09-14")].index
        all_pitches.loc[strange_games_LXI, "home_team"] = "SEA"
        all_pitches.loc[strange_games_LXI, "away_team"] = "OAK"

        strange_games_LXII = all_pitches[(all_pitches.home_team == "STL") & (
            all_pitches.away_team == "MIL") & (all_pitches.game_date == "2020-09-16")].index
        all_pitches.loc[strange_games_LXII, "home_team"] = "MIL"
        all_pitches.loc[strange_games_LXII, "away_team"] = "STL"

        strange_games_LXIII = all_pitches[(all_pitches.home_team == "TB") & (
            all_pitches.away_team == "BAL") & (all_pitches.game_date == "2020-09-17")].index
        all_pitches.loc[strange_games_LXIII, "home_team"] = "BAL"
        all_pitches.loc[strange_games_LXIII, "away_team"] = "TB"

        strange_games_LXIV = all_pitches[(all_pitches.home_team == "WSH") & (
            all_pitches.away_team == "MIA") & (all_pitches.game_date == "2020-09-18")].index
        all_pitches.loc[strange_games_LXIV, "home_team"] = "MIA"
        all_pitches.loc[strange_games_LXIV, "away_team"] = "WSH"

        strange_games_LXV = all_pitches[(all_pitches.home_team == "TOR") & (
            all_pitches.away_team == "PHI") & (all_pitches.game_date == "2020-09-18")].index
        all_pitches.loc[strange_games_LXV, "home_team"] = "PHI"
        all_pitches.loc[strange_games_LXV, "away_team"] = "TOR"

        strange_games_LXVI = all_pitches[(all_pitches.home_team == "STL") & (
            all_pitches.away_team == "PIT") & (all_pitches.game_date == "2020-09-18")].index
        all_pitches.loc[strange_games_LXVI, "home_team"] = "PIT"
        all_pitches.loc[strange_games_LXVI, "away_team"] = "STL"

        strange_games_LXVII = all_pitches[(all_pitches.home_team == "WSH") & (
            all_pitches.away_team == "MIA") & (all_pitches.game_date == "2020-09-20")].index
        all_pitches.loc[strange_games_LXVII, "home_team"] = "MIA"
        all_pitches.loc[strange_games_LXVII, "away_team"] = "WSH"

        strange_games_LXVIII = all_pitches[(all_pitches.home_team == "PHI") & (
            all_pitches.away_team == "WSH") & (all_pitches.game_date == "2020-09-22")].index
        all_pitches.loc[strange_games_LXVIII, "home_team"] = "WSH"
        all_pitches.loc[strange_games_LXVIII, "away_team"] = "PHI"

        strange_games_LXIX = all_pitches[(all_pitches.home_team == "COL") & (
            all_pitches.away_team == "ARI") & (all_pitches.game_date == "2020-09-25")].index
        all_pitches.loc[strange_games_LXIX, "home_team"] = "ARI"
        all_pitches.loc[strange_games_LXIX, "away_team"] = "COL"

        strange_games_LXX = all_pitches[(all_pitches.home_team == "SD") & (
            all_pitches.away_team == "SF") & (all_pitches.game_date == "2020-09-25")].index
        all_pitches.loc[strange_games_LXX, "home_team"] = "SF"
        all_pitches.loc[strange_games_LXX, "away_team"] = "SD"

        strange_games_LXXI = all_pitches[(all_pitches.home_team == "MIL") & (
            all_pitches.away_team == "STL") & (all_pitches.game_date == "2020-09-25")].index
        all_pitches.loc[strange_games_LXXI, "home_team"] = "STL"
        all_pitches.loc[strange_games_LXXI, "away_team"] = "MIL"

        strange_games_LXXII = all_pitches[(all_pitches.home_team == "SEA") & (
            all_pitches.away_team == "OAK") & (all_pitches.game_date == "2020-09-26")].index
        all_pitches.loc[strange_games_LXXII, "home_team"] = "OAK"
        all_pitches.loc[strange_games_LXXII, "away_team"] = "SEA"

        strange_games_LXXIII = all_pitches[(all_pitches.home_team == "NYM") & (
            all_pitches.away_team == "WSH") & (all_pitches.game_date == "2020-09-26")].index
        all_pitches.loc[strange_games_LXXIII, "home_team"] = "WSH"
        all_pitches.loc[strange_games_LXXIII, "away_team"] = "NYM"

        strange_games_LXXIV = all_pitches[(all_pitches.home_team == "SEA") & (
            all_pitches.away_team == "SF") & (all_pitches.game_date == "2020-09-16")].index
        all_pitches.loc[strange_games_LXXIV, "home_team"] = "SF"
        all_pitches.loc[strange_games_LXXIV, "away_team"] = "SEA"

        strange_games_LXXV = all_pitches[(all_pitches.home_team == "SEA") & (
            all_pitches.away_team == "SF") & (all_pitches.game_date == "2020-09-17")].index
        all_pitches.loc[strange_games_LXXV, "home_team"] = "SF"
        all_pitches.loc[strange_games_LXXV, "away_team"] = "SEA"

        strange_games_LXXVI = all_pitches[(all_pitches.home_team == "SEA") & (
            all_pitches.away_team == "SD") & (all_pitches.game_date == "2020-09-18")].index
        all_pitches.loc[strange_games_LXXVI, "home_team"] = "SD"
        all_pitches.loc[strange_games_LXXVI, "away_team"] = "SEA"

        strange_games_LXXVII = all_pitches[(all_pitches.home_team == "SEA") & (
            all_pitches.away_team == "SD") & (all_pitches.game_date == "2020-09-19")].index
        all_pitches.loc[strange_games_LXXVII, "home_team"] = "SD"
        all_pitches.loc[strange_games_LXXVII, "away_team"] = "SEA"

        strange_games_LXXVIII = all_pitches[(all_pitches.home_team == "SEA") & (
            all_pitches.away_team == "SD") & (all_pitches.game_date == "2020-09-20")].index
        all_pitches.loc[strange_games_LXXVIII, "home_team"] = "SD"
        all_pitches.loc[strange_games_LXXVIII, "away_team"] = "SEA"

        strange_games_LXXXI = all_pitches[(all_pitches.home_team == "WSH") & (all_pitches.away_team == "TOR") & (
            (all_pitches.game_date == "2021-04-27") | (all_pitches.game_date == "2021-04-27"))].index
        all_pitches.loc[strange_games_LXXXI, "home_team"] = "TOR"
        all_pitches.loc[strange_games_LXXXI, "away_team"] = "WSH"

        strange_games_LXXIX = all_pitches[(all_pitches.home_team == "TOR") & (
            all_pitches.away_team == "LAA") & (all_pitches.game_date == "2021-08-10")].index
        all_pitches.loc[strange_games_LXXIX, "home_team"] = "LAA"
        all_pitches.loc[strange_games_LXXIX, "away_team"] = "TOR"

        strange_games_LXXX = all_pitches[(all_pitches.home_team == "OAK") & (
            all_pitches.away_team == "DET") & (all_pitches.game_date == "2022-05-10")].index
        all_pitches.loc[strange_games_LXXX, "home_team"] = "DET"
        all_pitches.loc[strange_games_LXXX, "away_team"] = "OAK"

        return all_pitches