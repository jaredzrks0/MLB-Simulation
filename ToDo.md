See if we can attatch weather back to final dataset quicker (based on the index)
See if we can calculate league average info quicker in last step
See if we can vectorize data rolling with np convolve





# Building Datasets
## Things to Do:
- make sure all functions in build_datasets have descriptions.
- save a smaller (2016) and larger (2016-2018) training set to google cloud.
- Determine if program speed is an issue, and if so fix it!
- Add in current game situation (inning, score diff, runners on)
- Add in num batters faced for pitcher in game or inning (also maybe a rest metric like days since last pitched if its easy)
- Figure out why unnamed cols get added every pull and push from google cloud and stop that from happening (takes up data space)
- When pulling full weather we have an except statement that just returns a standard dome weather. We should cross ref with a list of domed stadiums to check how much this triggers to ensure its not too much, othwewise fix the underlying issue triggering the except (bad pybaseball home/away team labeling). Additionally, if we keep this command, it should probably be a little hotter than 72, to best reflect the average outdoor game (summer!)
  
## Ideas
- Right now rolling on a season and a month, but can exp with other time periods like 2 or 3 months, or 3, 7 games.
- I wonder if we will potentially overcount whatever the most recent play(s) for individual players are, as we roll by at bats rather than rolling at bats every game or even week. For example, if you do play x, particularly in a smaller sample, the percentage immedietly jumps and slowly lowers until it happens again. Additionally, when simulating we (should) update percentages with each at bat in game if we do this in training... 
- Make it easy to train different datasets on different rolling periods
  - In a similar vein, is it a problem that the rolling columns are likely super complicated? If so we use make proxy columns for the differences between them, but I feel like theres correlation again if we have more than 3 different rolling periods.
- VECTORIZE!!

# Game Simulation

## Ideas
- I wonder if we might undercount runs due to not all errors being counted as such. We could calculate the odds of runners advancing on things like out and fly out (non3b) to stop this.

# Other Models 

## To Do
- Build the models! We can wait until working on or finished with a basic simulation though and then incorporate these after if needed.
- Thinking of models for (Stealing, Sac Fly/Bunt, 1st -> 3rd, Balk??, Wild Pitch/Passed Ball)

## Ideas
- For stealing it is a bit harder bc it doesn't show up as a separate play, but can access in the pitch description with 'pitches[pitches.des.str.contains("steal") == True].iloc[40]'
- For the others, try using the descriptions too if the play is trickly and doesn't show up as an actual play.

# Other Things
- Make more OOP?
- Develop a way to account for new to the league players
- Why wont it recognize the cloud functions imports