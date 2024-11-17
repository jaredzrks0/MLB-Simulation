# Building Datasets
## Things to Do:
- Add in current game situation (inning, score diff, runners on)
- Add in num batters faced for pitcher in game or inning (also maybe a rest metric like days since last pitched if its easy)
- When pulling full weather we have an except statement that just returns a standard dome weather. We should cross ref with a list of domed stadiums to check how much this triggers to ensure its not too much, othwewise fix the underlying issue triggering the except (bad pybaseball home/away team labeling). Additionally, if we keep this command, it should probably be a little hotter than 72, to best reflect the average outdoor game (summer!)
- Right now if the player does not have the rolling window number of PA, we just roll what they have. We should do an analysis to impute either 'new' player stats or 'replacement level' stats, etc.
  
## Ideas
- I wonder if we will potentially overcount whatever the most recent play(s) for individual players are, as we roll by at bats rather than rolling at bats every game or even week. For example, if you do play x, particularly in a smaller sample, the percentage immedietly jumps and slowly lowers until it happens again. Additionally, when simulating we (should) update percentages with each at bat in game if we do this in training... 


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