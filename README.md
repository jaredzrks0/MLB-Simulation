This is a project to simulate MLB Games. Currently, the Pitch and Weather collection file pulls pitch by pitch MLB data and weather data for relevant games for use in the Building Datasets file. 
The Building Datasets file is currently being updated, but cleans data from the pitch/weather collection before calculating weather and park factors for each play/game and neutralizing the dataset for model testing in the Training Models file.
Finally, the Training Models file takes a preliminary look at models for predicting the outcome of PAs, with promising models including SVC (barring size restrictions), Logistic Regression, and a Neural Network, each with Log Loss < Base Case.

Future work will include expanding PA prediction to game decision simulation!
