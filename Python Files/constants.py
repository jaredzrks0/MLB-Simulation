#####################################################################
# Define constants for use in the building of training datasets
#####################################################################

HAND_COMBOS = ["RR", "RL", "LR", "LL"]
PLAY_TYPES = ['strikeout', 'fly_out', 'double', 'out', 'fielders_choice','error', 'walk', 'home_run',
          'single', 'sacrifice', 'double_play', 'intent_walk', 'triple']

# Team name and abbv conversions for use in attatching weather to pitches
WEATHER_NAME_CONVERSIONS = {"SF":"San Francisco Giants", "NYY":"New York Yankees", "DET":"Detroit Tigers", "TEX":"Texas Rangers",
                            "STL":"St. Louis Cardinals", "WSH":"Washington Nationals", "MIL":"Milwaukee Brewers", "CLE":"Cleveland Guardians",
                            "SD":"San Diego Padres", "COL":"Colorado Rockies", "BAL":"Baltimore Orioles", "HOU":"Houston Astros",
                            "KC":"Kansas City Royals", "OAK":"Oakland Athletics", "BOS":"Boston Red Sox", "CWS":"Chicago White Sox",
                            "AZ":"Arizona Diamondbacks","ARI":"Arizona Diamondbacks", "ATL":"Atlanta Braves", "CIN":"Cincinnati Reds", "MIN":"Minnesota Twins",
                            "MIA":"Miami Marlins", "LAD":"Los Angeles Dodgers", "TB":"Tampa Bay Rays", "PHI":"Philadelphia Phillies",
                            "NYM":"New York Mets", "CHC":"Chicago Cubs", "TOR":"Toronto Blue Jays", "SEA":"Seattle Mariners",
                            "LAA":"Los Angeles Angels", "PIT":"Pittsburgh Pirates"}      

# For filtering columns of raw statcast data
RELEVANT_BATTING_COLUMNS = ["game_date", "player_name", "batter", "pitcher", "events", "stand", "p_throws", "home_team", "away_team",
                            "hit_location", "bb_type", "on_3b", "on_2b", "on_1b", "outs_when_up", "inning", "inning_topbot","game_type",
                            "game_pk", "estimated_ba_using_speedangle", "launch_speed_angle", "bat_score", "fld_score", "post_bat_score",
                            "if_fielding_alignment", "of_fielding_alignment", "delta_home_win_exp"]

# For filtering rows of raw statcast data
RELEVANT_PLAY_TYPES = ["field_out", "strikeout", "strikeout_double_play", "force_out", "grounded_into_double_play", "double_play", "fielders_choice",
                    "fielders_choice_out", "other_out", "sac_fly", "sac_bunt", "single", "double", "triple", "home_run", 
                    "walk", "hit_by_pitch", "intent_walk", "field_error"]

# For converting/combining play title syntax of raw statcast data
PLAY_TYPE_DICT = {"field_out":"fly_out", "strikeout":"strikeout", "strikeout_double_play":"strikeout", "force_out":"out", "grounded_into_double_play":"double_play", "double_play":"double_play", "fielders_choice":"fielders_choice",
                    "fielders_choice_out":"fielders_choice", "other_out":"out", "sac_fly":"sacrifice", "sac_bunt":"sacrifice", "single":"single", "double":"double", "triple":"triple", "home_run":"home_run", 
                    "walk":"walk", "hit_by_pitch":"walk", "intent_walk":"intent_walk", "field_error":"error"}