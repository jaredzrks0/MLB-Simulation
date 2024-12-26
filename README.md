# MLB Game Simulation Project

This project simulates Major League Baseball (MLB) games by using a monte-carlo simulation to simulate individual plate appearances. The ultimate goal is to predict the outcomes of entire games, both in final score, but also expected stat lines for each player.

## Project Overview

The simulation process involves several key steps:

1. **Data Collection**: Gathering detailed MLB pitch-by-pitch data and corresponding weather information for each game in a training set.
2. **Data Preparation**: Cleaning and processing the collected dataset, calculating weather and park factors for each play and game, neutralizing the dataset for model testing, and calculating rolling averages of prior stats for different time periods.
3. **Model Training**: Developing and evaluating predictive models to forecast the outcomes of PAs. After testing, the strongest performing model is an XGBoost classifier trained on rolling average windows of [10, 40, 75, ~500] PAs. The model demonstrates performance with Log Loss below the base case in testing.
4. **Game Simulation**: The aforementioned trained model is then leveraged, alongside external information like daily team lineups and weather forecasts to simulate each MLB game of the day by predicting individual plate appearances in a Monte-Carlo simulation. *NOTE: This step is still in production*

## Future Work

Future enhancements for this project include:

- **Expanding Predictions**: Extending the current PA outcome model to simulate entire game decisions, providing a comprehensive game simulation experience.
- **User Interface Development**: Creating an intuitive interface for users view game simulations and player stats distributions.
- **Connection to Betting Lines**: Comparing predicted game outcomes and player stats distributions against live betting lines to identify positive value bets.

## Major Files and Their Usage

- **`get_lineups.py`**: Responsible for collecting pitch-by-pitch MLB data and corresponding weather information for relevant games.
- **`build_datasets/`**: Contains scripts for cleaning the collected data, calculating weather and park factors, and preparing neutralized datasets for underlying PA model testing.
- **`train_models/`**: Includes scripts that develop and evaluate predictive models for forecasting PA outcomes.
- **`simulate_games/`**: (Planned for future development) Will contain scripts to simulate entire MLB games based on the PA model.
- **`run.sh`**: A shell script to execute the entire simulation pipeline from daily data collection to game simulation.

## Getting Started

To set up the project environment and run the simulation pipeline:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/jaredzrks0/MLB-Simulation.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd MLB-Simulation
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Simulation Pipeline**:

   *NOTE: This step is still in production*

   ```bash
   ./run.sh
   ```

This will execute the data collection, preparation, and model training steps sequentially.

## Contributing

Contributions to enhance the project are welcome. Please refer to the `ToDo.md` file for a list of planned tasks and open issues. Feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For more information and to access the project repository, visit the [MLB Game Simulation GitHub page](https://github.com/jaredzrks0/MLB-Simulation).
