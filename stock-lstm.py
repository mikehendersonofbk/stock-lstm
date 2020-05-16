import requests
import io
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import backend as K
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from dotenv import load_dotenv
from config import Config

def download_daily_data(config, stock):
    url = '{}/query?function=TIME_SERIES_DAILY&symbol={}&datatype=csv&outputsize=full&apikey={}'.format(
        config.av_url,
        stock,
        config.av_api_key,
    )
    s=requests.get(url).content
    data = pd.read_csv(io.StringIO(s.decode('utf-8')))
    return data.set_index('timestamp').sort_index()

def augment_with_emas(df, emas):
    for i in emas:
        df['ema{}'.format(i)] = df['close'].ewm(span=i, min_periods=i).mean()
    return df

def append_lookback(data, num):
    df_s = data.copy(deep=True)
    for i in range(1, num):
        df = data.shift(i)
        cols = data.columns
        for col in cols:
            df_s['{}_t-{}'.format(col, i)] = df[col]
    return df_s

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = data.values
    scaled = scaler.fit_transform(values)

    data_scaled = pd.DataFrame(scaled, columns=data.columns, index=data.index)
    return data_scaled

def create_model():
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

if __name__ == '__main__':
    config = Config()
    daily_data = augment_with_emas(download_daily_data(config, 'MSFT'), [9, 12, 20, 50, 100])
    daily_data.fillna(0, inplace=True)

    daily_data_supervised = append_lookback(daily_data, config.lookback)

    # isolate the 'y'
    close_df = daily_data_supervised.pop('close')
    daily_data_supervised['close'] = close_df

    # scale data
    daily_data_supervised = scale_data(daily_data_supervised)
    print(daily_data_supervised.head(50))

    train_size = math.ceil(daily_data_supervised.shape[0] * .8) # 80% of data for training
    train = daily_data_supervised.iloc[:train_size, :]
    test = daily_data_supervised.iloc[train_size:, :]

    train_set = train.values
    test_set = test.values

    train_X, train_y = train_set[:, :-1], train_set[:, -1]
    test_X, test_y = test_set[:, :-1], test_set[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = create_model()
    history = model.fit(train_X, train_y, epochs=50, validation_data=(test_X, test_y), verbose=1, shuffle=False, batch_size=64)
    
    # save model
    model.save('output-model')