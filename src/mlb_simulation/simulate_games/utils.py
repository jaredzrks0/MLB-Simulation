def convert_wind_direction(wind_direction):
    if wind_direction == 'L-R':
        return 'left to right'
    elif wind_direction == 'R-L':
        return 'right to left'
    elif wind_direction == 'Out':
        return wind_direction.lower()
    else:
        return wind_direction

def convert_rotowire_weather_to_proference(weather):
    storage = {}
    home_team = weather.game_id.split('@ ')[-1].split(' on')[0]
    wind_direction = convert_wind_direction(weather.wind_direction[0])

    if weather.is_dome[0] == False:
        rain_percentage = weather.rain_percentage[0]
        temprature_sq = weather.temprature[0] ** 2
        wind_speed = weather.wind_speed[0]

        storage['home_team'] = home_team
        storage['rain_percentage'] = rain_percentage
        storage['temprature_sq'] = temprature_sq
        storage['out'] = 0
        storage['in'] = 0
        storage['zero'] = 0
        storage['right to left'] = 0
        storage['left to right'] = 0
        storage[wind_direction] = wind_speed

    else:
        rain_percentage = 0
        temprature_sq = 71 ** 2
        wind_speed = 0

        storage['home_team'] = home_team
        storage['rain_percentage'] = rain_percentage
        storage['temprature_sq'] = temprature_sq
        storage['out'] = 0
        storage['in'] = 0
        storage['zero'] = 0
        storage['Right to Left'] = 0
        storage['Left to Right'] = 0

    return storage