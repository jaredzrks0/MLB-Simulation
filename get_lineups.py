import requests
import pickle


from bs4 import BeautifulSoup as bs


def mlb_scrape(date):
    url=f'https://www.mlb.com/starting-lineups/{date}'
    response = requests.get(url)
    soup = bs(response.content, features="lxml")

    lineups = {}
    days_games = []

    all_games = soup.find_all('div', {'class':'starting-lineups__matchup'})

    i = 0
    for game in all_games:
        lineups[i] = {}

        location = game.find_all('div', {'class':'starting-lineups__game-location'})[0].text.strip()

        home_team = game.find_all('span', {'class': 'starting-lineups__team-name starting-lineups__team-name--home'})[0].text.split()[0]
        lineups[i]['home_team'] = home_team if home_team not in ["Red", "White"] else home_team + "Sox"
        away_team = game.find_all('span', {'class': 'starting-lineups__team-name starting-lineups__team-name--away'})[0].text.split()[0]
        lineups[i]['away_team'] = away_team if home_team not in ["Red", "White"] else home_team + "Sox"

        home_lineup = game.find_all('ol', {'class':'starting-lineups__team starting-lineups__team--home'})[0]
        away_lineup = game.find_all('ol', {'class':'starting-lineups__team starting-lineups__team--away'})[0]

        home_pitcher = game.find_all('a', {'class':'starting-lineups__pitcher--link'})[3].text
        home_pitcher_id = game.find_all('a', {'class':'starting-lineups__pitcher--link'})[3]['href'].split('-')[-1]
        away_pitcher = game.find_all('a', {'class':'starting-lineups__pitcher--link'})[1].text
        away_pitcher_id = game.find_all('a', {'class':'starting-lineups__pitcher--link'})[1]['href'].split('-')[-1]

        home_players = home_lineup.find_all('li', {'class':'starting-lineups__player'})
        home_player_names = [player.a.text for player in home_players]
        home_player_positions = [player.span.text.split(") ")[-1].strip() for player in home_players]
        home_player_ids = [player.a['href'].split("-")[-1] for player in home_players]

        away_players = away_lineup.find_all('li', {'class':'starting-lineups__player'})
        away_player_names = [player.a.text for player in away_players]
        away_player_positions = [player.span.text.split(") ")[-1].strip() for player in away_players]
        away_player_ids = [player.a['href'].split("-")[-1] for player in away_players]

        # Insert into Dictionary and List
        lineups[i]['home_pitcher'] = {'name':home_pitcher, 'id':home_pitcher_id}
        lineups[i]['away_pitcher'] = {'name':away_pitcher, 'id':away_pitcher_id}
        lineups[i]['stadium'] = location

        lineups[i]['home_lineup'] = {n+1:{'position':home_player_positions[n], 'player':home_player_names[n], 'id':home_player_ids[n]} for n in range(len(home_player_positions))}
        lineups[i]['away_lineup'] = {n+1:{'position':away_player_positions[n], 'player':away_player_names[n], 'id':away_player_ids[n]} for n in range(len(away_player_positions))}

        days_games.append((home_team, away_team))
        i += 1

    return {'games':days_games, 'lineups':lineups}


def rotowire_scrape(url='https://www.rotowire.com/baseball/daily-lineups.php'):
    response = requests.get(url)
    soup = bs(response.content, 'html')

    lineups = {}

    all_games = soup.find_all('div', {'class':'lineup is-mlb'})

    i = 0
    for game in all_games:
        lineups[i] = {}

        home_team = soup.find_all('div', {'class':'lineup__mteam is-home'})[i].text.split()[0]
        lineups[i]['home_team'] = home_team
        away_team = soup.find_all('div', {'class':'lineup__mteam is-visit'})[i].text.split()[0]
        lineups[i]['away_team'] = away_team

        home_lineup = soup.find_all('ul', {'class':'lineup__list is-home'})[i]
        home_pitcher = home_lineup.find_all('div', {'class':'lineup__player-highlight-name'})[0].text.strip().split("\n")[0]
        home_players = home_lineup.find_all('li', {'class':'lineup__player'})
        home_positions = [home_players[n].div.text for n in range(len(home_players))]
        home_names = [home_players[n].a['title'] for n in range(len(home_players))]

        if home_lineup.find_all('li', {'class':'lineup__status is-confirmed'}):
            home_confidence = "Confirmed"
        elif home_lineup.find_all('li', {'class':'lineup__status is-expected'}):
            home_confidence = "Expected"
        else:
            home_confidence = "Unknown"
        
        away_lineup = soup.find_all('ul', {'class':'lineup__list is-visit'})[i]
        away_pitcher = away_lineup.find_all('div', {'class':'lineup__player-highlight-name'})[0].text.strip().split("\n")[0]
        away_players = away_lineup.find_all('li', {'class':'lineup__player'})
        away_positions = [away_players[n].div.text for n in range(len(away_players))]
        away_names = [away_players[n].a['title'] for n in range(len(away_players))]

        if away_lineup.find_all('li', {'class':'lineup__status is-confirmed'}):
            away_confidence = "Confirmed"
        elif away_lineup.find_all('li', {'class':'lineup__status is-expected'}):
            away_confidence = "Expected"
        else:
            away_confidence = "Unknown"

        # Insert into our dictionary
        lineups[i]['home_confidence'] = home_confidence
        lineups[i]['away_confidence'] = away_confidence

        lineups[i]['home_pitcher'] = home_pitcher
        lineups[i]['away_pitcher'] = away_pitcher

        lineups[i]['home_lineup'] = {n+1:{'position':home_positions[n], 'player':home_names[n]} for n in range(len(home_positions))}
        lineups[i]['away_lineup'] = {n+1:{'position':away_positions[n], 'player':away_names[n]} for n in range(len(away_positions))}

        i += 1

    return lineups
