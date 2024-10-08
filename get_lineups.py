import requests
from bs4 import BeautifulSoup as bs

def scrape_lineups(url='https://www.rotowire.com/baseball/daily-lineups.php'):
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


