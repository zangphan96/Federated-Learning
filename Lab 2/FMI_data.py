import pandas as pd


# Input: hourly air temperature measurements for the single month duration
# Output: daily average daytime air temperature for the single month duration
def daily_avg(df):
    # Daytime is defined as time interval from 12:00 to 18:00
    daytime_df = df[df['Time'].isin(['12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00'])]

    daytime_avg_df = pd.DataFrame({'Average air temperature': [], 'd': [], 'm': [], 'y': []})
    for i in range(0, len(daytime_df) - 6, 7):
        avg_temp = (float(daytime_df.iloc[i]['Air temperature (degC)']) + float(
            daytime_df.iloc[i + 1]['Air temperature (degC)']) +
                    float(daytime_df.iloc[i + 2]['Air temperature (degC)']) + float(
                    daytime_df.iloc[i + 3]['Air temperature (degC)']) +
                    float(daytime_df.iloc[i + 4]['Air temperature (degC)']) + float(
                    daytime_df.iloc[i + 5]['Air temperature (degC)']) +
                    float(daytime_df.iloc[i + 6]['Air temperature (degC)'])) / 7

        d = daytime_df.iloc[i]['d']
        m = daytime_df.iloc[i]['m']
        y = daytime_df.iloc[i]['Year']
        daytime_avg_df.loc[len(daytime_avg_df.index)] = [avg_temp, d, m, y]

    return daytime_avg_df

