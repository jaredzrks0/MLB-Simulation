import pandas as pd
import numpy as np
import pickle as pkl
import warnings
import datetime
import numpy as np
from numpy import random
from datetime import datetime as dt
import datetime
from get_lineups import mlb_scrape

from build_datasets.dataset_builder import DatasetBuilder
from utils import convert_rotowire_weather_to_proference


warnings.simplefilter('ignore')


class GameSimulation():
    def __init__(self, date, home_team, lineup_dict, PA_model, encoder, verbose=False):
        self.date = date
        self.year, self.month, self.day = str(self.date.year), str(self.date.month), str(self.date.day)
        self.PA_model = PA_model
        self.encoder = encoder
        self.verbose = verbose
        self.lineup_dict = lineup_dict
        self.home_lineup = self.lineup_dict['home_lineup']
        self.away_lineup = self.lineup_dict['away_lineup']
        self.home_pitcher = self.lineup_dict['home_pitcher']['id']
        self.away_pitcher = self.lineup_dict['away_pitcher']['id']

        # Grab the daily rolled stats
        with open(f'../../../../MLB-Data/daily_stats_dfs/daily_stats_df_updated_{self.year.zfill(2)}-{self.month.zfill(2)}-{self.day.zfill(2)}.pkl', 'rb') as fpath:
            self.daily_dataset = pkl.load(fpath)
            self.daily_dataset = self.daily_dataset.drop(columns = ['play_type', 'is_on_base'])

        # Grab the expected weather data
        with open(f'../../../../MLB-Data/rotowire_weather_data/weather_data_updated_{self.year.zfill(2)}-{self.month.zfill(2)}-{self.day.zfill(2)}', 'rb') as fpath:
            self.expected_weather = pkl.load(fpath)

        # Grab the team name conversions for the weather conversion
        self.name_conversions = pd.read_excel('../build_datasets/data/non_mlb_data/Ballpark Info.xlsx', header=2)

        # Get the individual stats for each batter and pitcher + the league at the given moment, so we can pull them as needed for a given at bat
        batter_stats = self.daily_dataset.groupby(by='batter').last()
        batter_stats = batter_stats[[col for col in batter_stats.columns if 'PA' in col and 'LA' not in col and 'pitcher' not in col]]

        pitcher_stats = self.daily_dataset.groupby(by='pitcher').last()
        pitcher_stats = pitcher_stats[[col for col in pitcher_stats.columns if 'PA' in col and 'LA' not in col and 'pitcher' in col]]

        LA_stats = self.daily_dataset.iloc[-1] #Just pull the final row bc we know that all the LA columns are the same for the given day
        LA_stats = LA_stats[[col for col in LA_stats.index if 'LA' in col]]

        # Turn the stats into dicts for faster access
        self.batter_stats = dict(batter_stats.T)
        self.pitcher_stats = dict(pitcher_stats.T)
        self.LA_stats = pd.Series(LA_stats)

        # Finally, Initialize the game state
        self.home_team = home_team
        self.home_park = self.name_conversions[self.name_conversions['Full Name'] == self.home_team].Stadium.iloc[0]
        self.inning = 1
        self.inning_topbot = 1 # 1 for top 0 for bottom
        self.pitbat = None
        self.on_3b = 0
        self.on_2b = 0
        self.on_1b = 0
        self.outs_when_up = 0
        self.bat_score = 0
        self.field_score = 0
        self.score_tracker = {'home':0, 'away':0}
        
        # Initialize Weather info
        self.weather_row = self.expected_weather[self.expected_weather.game_id.str.contains(self.home_team)].iloc[0] # Just grabs the first game - this 'fails' if 2x header
        self.converted_weather = convert_rotowire_weather_to_proference(self.weather_row)

        # Initialize handedness dictionaries
        self.batter_handedness = self.daily_dataset.groupby('batter')['pitbat'].apply(lambda x: x.iloc[-1][0]).to_dict()
        self.pitcher_handedness = self.daily_dataset.groupby('pitcher')['pitbat'].apply(lambda x: x.iloc[-1][1]).to_dict()
    
    def _get_pitbat(self, batter_id, pitcher_id):
        batter_hand = self.batter_handedness.get(batter_id, "X")
        pitcher_hand = self.pitcher_handedness.get(pitcher_id, "X")
        return batter_hand + pitcher_hand
    
    def make_PA_row(self, batter_id, pitcher_id):
        data = {
            'ballpark': self.home_park,
            'batter': batter_id,
            'pitcher': pitcher_id,
            'pitbat': self._get_pitbat(batter_id, pitcher_id),
            'on_3b': self.on_3b,
            'on_2b': self.on_2b,
            'on_1b': self.on_1b,
            'outs_when_up': self.outs_when_up,
            'inning': self.inning,
            'inning_topbot': self.inning_topbot,
            'bat_score': self.bat_score,
            'fld_score': self.field_score
            }

        series = pd.Series(data)

        # Insert the batting, pitching, and LA stats
        batter_stats = self.batter_stats[batter_id]
        pitcher_stats = self.pitcher_stats[pitcher_id]
        LA_stats = self.LA_stats

        # Insert the weather data
        weather_dict = self.converted_weather
        weather_series = pd.Series(weather_dict)
        series = pd.concat([series, batter_stats, pitcher_stats, LA_stats, weather_series])
        self.current_PA = series.to_frame().T
        
        # # Make sure all the columns are in the original order
        self.current_PA = self.current_PA[self.daily_dataset.columns]
           
    def update_current_PA(self, batter_id, pitcher_id):
        batter_stats = self.batter_stats[batter_id]
        pitcher_stats = self.pitcher_stats[pitcher_id]

        # Create a new temporary series with the batter stats and pitcher stats
        batter_stats.loc[batter_stats.index] = batter_stats
        pitcher_stats.loc[pitcher_stats.index] = pitcher_stats
    
    def predict_PA(self):
        probabilities = self.PA_model.predict_proba(self.current_PA).flatten()
        outcome = np.random.choice(self.encoder.categories_[0], p=probabilities)
        return outcome
       

    def simulate_game(self):
        # Define which batter is currently batting in the lineup
        self.lineup_tracker = {'home':1, 'away':1}
        self.inning = 1
        self.on_1b = 0
        self.on_2b = 0
        self.on_3b = 0
        self.score_tracker = {'home':0, 'away':0}

        # Start with away team batting
        while self.inning <= 1:
            # Start the away team's half inning
            self.batting_team = 'away'
            self.inning_topbot = 1
            self.bat_score = self.score_tracker['away']
            self.field_score = self.score_tracker['home']
            if self.verbose:
                print(f"Inning {self.inning}, Away Team")
            self.simulate_inning(self.away_lineup, self.batting_team)
            
            # If it's not the 9th inning, switch to the home team's turn
            if self.inning < 9:
                self.batting_team = 'home'
                self.inning_topbot = 0
                self.bat_score = self.score_tracker['home']
                self.field_score = self.score_tracker['away']
                if self.verbose:
                    print(f"Inning {self.inning}, Home Team")
                self.simulate_inning(self.home_lineup, self.batting_team)
            
            self.inning += 1
            if self.verbose:
                print(f"Score after inning {self.inning - 1}: Away - {self.score_tracker['away']}, Home - {self.score_tracker['home']}\n")

    def simulate_inning(self, lineup, team_type):
        # Track outs in an inning
        self.outs_when_up = 0

        while self.outs_when_up < 3:  # Three outs in an inning
            batter_id = float(lineup[self.lineup_tracker[team_type]]['id'])
            pitcher_id = float(self.home_pitcher if team_type == 'away' else self.away_pitcher)

            # Create the at-bat row for the batter and pitcher
            self.make_PA_row(batter_id, pitcher_id)
            
            # Predict the outcome of the at-bat
            outcome = self.predict_PA()

            # Handle the outcome accordingly
            self.handle_outcome(outcome, team_type)
            
            # Move to the next batter (circular lineup)
            self.lineup_tracker[team_type] = (self.lineup_tracker[team_type] + 1) % 9
            self.lineup_tracker[team_type] = self.lineup_tracker[team_type] + 1 if self.lineup_tracker[team_type] == 0 else self.lineup_tracker[team_type]

        # Reset the game state
        self.outs_when_up = 0
        self.on_1b = 0
        self.on_2b = 0
        self.on_3b = 0
        
    def handle_outcome(self, outcome, team_type):
        if outcome == 'strikeout':
            self.outs_when_up += 1  # Increment outs when batter is out
        elif outcome == 'field_out':
            self.outs_when_up += 1  # Increment outs when batter is out
        elif outcome == 'walk':
            if self.on_1b == 0: # Bases Empty
                self.on_1b == 1
            elif self.on_2b == 0: # Just man on 1b
                self.on_2b == 1
            elif self.on_3b == 0: # Men on 1b and 2b
                self.on_3b == 1
            else: # Bases loaded
                self.score_tracker[team_type] += 1
        elif outcome == 'single':
            self.handle_base_hit(1)  # Handle single (advance to 1st base)
        elif outcome == 'double':
            self.handle_base_hit(2)  # Handle double (advance to 2nd base)
        elif outcome == 'home_run':
            self.handle_home_run()  # Handle home run (score and reset bases)
        elif outcome == 'error':
            # Handle error (place batter on base without advancing outs)
            error_value = np.random.random()
            if error_value > 0.75:
                self.handle_base_hit(2) # 2 base error with 25% chance
            else:
                self.handle_base_hit(1) # 1 base error with 75% chance
        elif outcome == 'double_play':
            self.outs_when_up += 2  # Double play = two outs
            if self.on_1b and self.on_2b:  # DP with runners on 1st and 2nd
                self.on_1b = 0  # Remove runner on 1st
                self.on_2b = 0  # Remove runner on 2nd
            elif self.on_2b and self.on_3b:  # DP with runners on 2nd and 3rd
                self.on_2b = 0  # Remove runner on 2nd
                self.on_3b = 0  # Remove runner on 3rd
            elif self.on_1b and self.on_3b:  # NOT adjacent, remove 1st base only
                self.on_1b = 0  # Remove runner on 1st
                if self.outs_when_up < 3: # Then the runner on 3b can score
                    self.on_3b = 0
                    self.score_tracker[team_type] += 1
            else:  # If only one runner is on base, remove them
                if self.on_3b:
                    self.on_3b = 0
                elif self.on_2b:
                    self.on_2b = 0
                elif self.on_1b:
                    self.on_1b = 0
        elif outcome == 'sacrifice':
            # Advance all runners by one base
            self.outs_when_up += 1
            if self.on_3b:
                self.on_3b = 0
                self.score_tracker[team_type] += 1
            if self.on_2b:
                self.on_2b = 0
                self.on_3b = 1
            if self.on_1b:
                self.on_1b = 0
                self.on_2b = 1
        elif outcome == 'triple':
            self.handle_base_hit(3)  # Handle triple (advance to 3rd base)

    def handle_base_hit(self, bases):
        # Handle moving runners based on the type of base hit (single, double, triple)
        if bases == 1:
            self.on_1b = 1  # Batter advances to 1st base
        elif bases == 2:
            self.on_2b = 1  # Batter advances to 2nd base
        elif bases == 3:
            self.on_3b = 1  # Batter advances to 3rd base

        # Move any existing runners on base
        self.advance_runners(bases)

    def advance_runners(self, bases):
        # Update base runners based on the type of hit
        # Update base runners based on the type of hit
        if bases == 1:  # Single
            if self.on_3b:
                self.score_tracker[self.batting_team] += 1  # Run scores from 3rd base
                self.on_3b = 0

            runner_on_2b_scores = self.on_2b and random.random() < 0.62  # 62% chance to score
            runner_on_1b_scores = self.on_1b and random.random() < 0.01  # 1% chance to score
            runner_on_1b_advances = self.on_1b and runner_on_2b_scores and random.random() < 0.40  # Can only advance if 2nd scores

            if runner_on_2b_scores:
                self.score_tracker[self.batting_team] += 1  # Score from 2nd
                self.on_2b = 0  # Clear 2nd base

            if runner_on_1b_scores:
                self.score_tracker[self.batting_team] += 1  # Score from 1st
                self.on_1b = 0  # Clear 1st base
            elif runner_on_1b_advances:
                self.on_3b = self.on_1b  # Move to 3rd
                self.on_1b = 0  # Clear 1st base
            else:
                self.on_2b = self.on_1b  # Normal advance to 2nd

            self.on_1b = 1  # Batter takes 1st base

        elif bases == 2:  # Double
            if self.on_3b:
                self.score_tracker[self.batting_team] += 1  # Run scores from 3rd base
                self.on_3b = 0

            if self.on_2b:
                self.score_tracker[self.batting_team] += 1  # Run scores from 2nd base
                self.on_2b = 0  # Clear 2nd base since they scored

            runner_on_1b_scores = self.on_1b and random.random() < 0.38  # 38% chance to score from 1st

            if runner_on_1b_scores:
                self.score_tracker[self.batting_team] += 1  # Runner from 1st scores
                self.on_1b = 0  # Clear 1st base
            else:
                self.on_3b = self.on_1b  # Move runner from 1st to 3rd

            self.on_2b = 1  # Batter takes 2nd base
            self.on_1b = 0  # 1st base remains empty after a double

        elif bases == 3:  # Triple
            if self.on_3b != 0:
                self.score_tracker[self.batting_team] += 1  # Run scores from 3rd base
            if self.on_2b != 0:
                self.score_tracker[self.batting_team] += 1  # Run scores from 2nd base
            if self.on_1b != 0:
                self.score_tracker[self.batting_team] += 1
            self.on_3b = 1
            self.on_2b = 0
            self.on_1b = 0

    def handle_home_run(self):
        # Home run: Score the batter and clear the bases
        self.score_tracker[self.batting_team] += sum([self.on_1b, self.on_2b, self.on_3b, 1])
        self.on_3b = 0
        self.on_2b = 0
        self.on_1b = 0
