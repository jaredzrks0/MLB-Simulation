import pandas as pd
import polars as pl

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
def _correct_home_away_swap(all_pitches: pl.LazyFrame) -> pl.LazyFrame:
    """Function used to correct a series of games across years where the home team and away team are swapped on baseball reference

    ------------INPUTS------------
    - all_pitches: pl.LazyFrame
        A LazyFrame of individual pitches, pulled from the statcast API.

    ------------OUTPUTS------------
    - all_pitches: pl.LazyFrame
        A LazyFrame of individual pitches, identical to the function's input, other than the correction of home and away teams in
        a select subset of games.
    """
    # Define a helper function to create expressions for swapping teams
    def swap_teams_expr(home_team, away_team, game_date=None):
        # Build filter condition
        condition = (pl.col("home_team") == home_team) & (pl.col("away_team") == away_team)
        if game_date:
            condition = condition & (pl.col("game_date") == game_date)
            
        # Create expressions for the update
        home_expr = pl.when(condition).then(away_team).otherwise(pl.col("home_team")).alias("home_team")
        away_expr = pl.when(condition).then(home_team).otherwise(pl.col("away_team")).alias("away_team")
        
        return [home_expr, away_expr]
    
    # Build a list of all expressions for team swaps
    swap_expressions = []
    
    # General swaps (no date)
    swap_expressions.extend(swap_teams_expr("TOR", "WSH"))
    
    # Dated swaps
    swap_expressions.extend(swap_teams_expr("CIN", "SF", "2013-07-23"))
    
    # BAL-TB swaps
    swap_expressions.extend(swap_teams_expr("BAL", "TB", "2015-05-01"))
    swap_expressions.extend(swap_teams_expr("BAL", "TB", "2015-05-02"))
    swap_expressions.extend(swap_teams_expr("BAL", "TB", "2015-05-03"))
    
    # MIA-MIL swaps
    swap_expressions.extend(swap_teams_expr("MIA", "MIL", "2017-09-15"))
    swap_expressions.extend(swap_teams_expr("MIA", "MIL", "2017-09-16"))
    swap_expressions.extend(swap_teams_expr("MIA", "MIL", "2017-09-17"))
    
    # 2020 season swaps
    swap_expressions.extend(swap_teams_expr("NYY", "PHI", "2020-08-05"))
    swap_expressions.extend(swap_teams_expr("MIA", "BAL", "2020-08-05"))
    swap_expressions.extend(swap_teams_expr("MIA", "BAL", "2020-08-06"))
    swap_expressions.extend(swap_teams_expr("MIA", "BAL", "2020-08-07"))
    swap_expressions.extend(swap_teams_expr("STL", "CHC", "2020-08-17"))
    swap_expressions.extend(swap_teams_expr("STL", "CHC", "2020-08-18"))
    swap_expressions.extend(swap_teams_expr("STL", "CHC", "2020-08-19"))
    swap_expressions.extend(swap_teams_expr("MIA", "WSH", "2020-08-22"))
    swap_expressions.extend(swap_teams_expr("MIA", "NYM", "2020-08-25"))
    swap_expressions.extend(swap_teams_expr("NYY", "ATL", "2020-08-26"))
    swap_expressions.extend(swap_teams_expr("CIN", "MIL", "2020-08-27"))
    swap_expressions.extend(swap_teams_expr("SEA", "SD", "2020-08-27"))
    swap_expressions.extend(swap_teams_expr("LAD", "SF", "2020-08-27"))
    swap_expressions.extend(swap_teams_expr("PIT", "STL", "2020-08-27"))
    swap_expressions.extend(swap_teams_expr("NYM", "NYY", "2020-08-28"))
    swap_expressions.extend(swap_teams_expr("MIN", "DET", "2020-08-29"))
    swap_expressions.extend(swap_teams_expr("OAK", "HOU", "2020-08-29"))
    swap_expressions.extend(swap_teams_expr("CHC", "CIN", "2020-08-29"))
    swap_expressions.extend(swap_teams_expr("NYM", "NYY", "2020-08-30"))
    swap_expressions.extend(swap_teams_expr("WSH", "ATL", "2020-09-04")) 
    swap_expressions.extend(swap_teams_expr("NYY", "BAL", "2020-09-04"))
    swap_expressions.extend(swap_teams_expr("TOR", "BOS", "2020-09-04"))
    swap_expressions.extend(swap_teams_expr("DET", "MIN", "2020-09-04"))
    swap_expressions.extend(swap_teams_expr("CIN", "PIT", "2020-09-04"))
    swap_expressions.extend(swap_teams_expr("HOU", "LAA", "2020-09-05"))
    swap_expressions.extend(swap_teams_expr("STL", "CHC", "2020-09-05"))
    swap_expressions.extend(swap_teams_expr("HOU", "OAK", "2020-09-08"))
    swap_expressions.extend(swap_teams_expr("BOS", "PHI", "2020-09-08"))
    swap_expressions.extend(swap_teams_expr("MIN", "STL", "2020-09-08"))
    swap_expressions.extend(swap_teams_expr("DET", "STL", "2020-09-10"))
    swap_expressions.extend(swap_teams_expr("PHI", "MIA", "2020-09-11"))
    swap_expressions.extend(swap_teams_expr("BAL", "NYY", "2020-09-11"))
    swap_expressions.extend(swap_teams_expr("OAK", "TEX", "2020-09-12"))
    swap_expressions.extend(swap_teams_expr("PHI", "MIA", "2020-09-13"))
    swap_expressions.extend(swap_teams_expr("SF", "SD", "2020-09-13"))
    swap_expressions.extend(swap_teams_expr("PIT", "CIN", "2020-09-14"))
    swap_expressions.extend(swap_teams_expr("STL", "MIL", "2020-09-14"))
    swap_expressions.extend(swap_teams_expr("OAK", "SEA", "2020-09-14"))
    swap_expressions.extend(swap_teams_expr("STL", "MIL", "2020-09-16"))
    swap_expressions.extend(swap_teams_expr("SEA", "SF", "2020-09-16"))
    swap_expressions.extend(swap_teams_expr("SEA", "SF", "2020-09-17"))
    swap_expressions.extend(swap_teams_expr("TB", "BAL", "2020-09-17"))
    swap_expressions.extend(swap_teams_expr("WSH", "MIA", "2020-09-18"))
    swap_expressions.extend(swap_teams_expr("TOR", "PHI", "2020-09-18"))
    swap_expressions.extend(swap_teams_expr("STL", "PIT", "2020-09-18"))
    swap_expressions.extend(swap_teams_expr("SEA", "SD", "2020-09-18"))
    swap_expressions.extend(swap_teams_expr("SEA", "SD", "2020-09-19"))
    swap_expressions.extend(swap_teams_expr("WSH", "MIA", "2020-09-20"))
    swap_expressions.extend(swap_teams_expr("SEA", "SD", "2020-09-20"))
    swap_expressions.extend(swap_teams_expr("PHI", "WSH", "2020-09-22"))
    swap_expressions.extend(swap_teams_expr("COL", "ARI", "2020-09-25"))
    swap_expressions.extend(swap_teams_expr("SD", "SF", "2020-09-25"))
    swap_expressions.extend(swap_teams_expr("MIL", "STL", "2020-09-25"))
    swap_expressions.extend(swap_teams_expr("SEA", "OAK", "2020-09-26"))
    swap_expressions.extend(swap_teams_expr("NYM", "WSH", "2020-09-26"))
    
    # 2021 season swaps
    swap_expressions.extend(swap_teams_expr("WSH", "TOR", "2021-04-27"))
    swap_expressions.extend(swap_teams_expr("TOR", "LAA", "2021-08-10"))
    
    # 2022 season swaps
    swap_expressions.extend(swap_teams_expr("OAK", "DET", "2022-05-10"))
    
    # Apply all the swap expressions in a single with_columns operation
    return all_pitches.with_columns(swap_expressions)
