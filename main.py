import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf

# Connect to database
host = "ec2-52-6-117-96.compute-1.amazonaws.com"
dbname = "dftej5l5m1cl78"
user = "aiuhlrpcnftsjs"
password = "8b2220cd5b6da572369545d91f6b435dfc37a42bfec6b6e2a5c9f236dfb65f42"

conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
cur = conn.cursor()

# define the stock symbol and the date range
symbol = "JPM"
end_date = datetime(2023, 1, 4)
start_date = datetime(2000, 1, 1)

# Pull data and format

query = "SELECT date, close, ema_200, ema_50, ema_12, ema_26, upper_band, lower_band, macd_results, stochastic_oscillator, rsi FROM daily_stock_price WHERE date >= %s AND date <= %s ORDER BY date;"
cur.execute(query, (start_date, end_date))
data = cur.fetchall()

data = pd.DataFrame(data, columns=['date', 'close', 'ema_200', 'ema_50', 'ema_12', 'ema_26', 'upper_band', 'lower_band', 'macd_results', 'stochastic_oscillator', 'rsi'])
data = data.sort_values(by='date', ascending=True)

X = data[['close', 'ema_200', 'ema_50', 'ema_12', 'ema_26', 'upper_band', 'lower_band', 'macd_results', 'stochastic_oscillator', 'rsi']].values

close_prices = data['close'].values
y = (close_prices - np.roll(close_prices, 5)) / np.roll(close_prices, 5)

y = y.reshape(-1, 1)
y = y[30:]
X = X[30:]


split_index = int(0.8 * len(X))
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]
print(len(X_test), len(y_test))

sequence_length = 100
batch_size = 1
train_data = tf.keras.preprocessing.sequence.TimeseriesGenerator(
    X_train,y_train,
    length=sequence_length,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=None,
    shuffle=False,
    reverse=False,
    batch_size=batch_size
)

filtered_train_data = []
for i in range(len(train_data)):
    x_sequence, y_sequence = train_data[i]
    if (y_sequence >= 0.015) or (y_sequence <= -0.015):
        filtered_train_data.append(train_data[i])

sequence_length = 100
batch_size = 1
test_data = tf.keras.preprocessing.sequence.TimeseriesGenerator(
    X_test,y_test,
    length=sequence_length,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=None,
    shuffle=False,
    reverse=False,
    batch_size=batch_size
)

# define the LSTM model
model = Sequential()
model.add(LSTM(150, input_shape=(100, 10)))  # LSTM layer with 64 units
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_checkpoint = ModelCheckpoint('stock_model_v4_v1.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

accumulated_X = []  
accumulated_y = []  

for batch, (X_train, y_train) in enumerate(filtered_train_data):
    y_train = np.where(y_train > 0, 1, 0)

    X_normalized = np.empty_like(X_train, dtype=np.float64)  
    for sequence in range(X_train.shape[0]):
        first_data_point = X_train[0, 0]  
        X_normalized = (X_train - first_data_point) / first_data_point
        X_train[:, :, :7] = X_normalized[:, :, :7]
    accumulated_X.append(X_train)
    accumulated_y.append(y_train)

np.set_printoptions(threshold=np.inf)
accumulated_X = np.vstack(accumulated_X)
accumulated_y = np.vstack(accumulated_y)

history = model.fit(accumulated_X, accumulated_y, batch_size=1, epochs=10)
accuracy = history.history['accuracy']
print(f"accuracy: {accuracy}")

model.save('stock_model_v4_v1.h5')

cur.execute("DROP TABLE IF EXISTS stock_predictions_v4")

cur.execute("CREATE TABLE IF NOT EXISTS stock_predictions_v4 (date REAL, actual_change REAL, predicted_change REAL)")

for batch, (X_test, y_test) in enumerate(test_data):
    y_test = np.where(y_test > 0, 1, 0)

    X_normalized = np.empty_like(X_test, dtype=np.float64)  
    for sequence in range(X_test.shape[0]):
        first_data_point = X_test[0, 0] 
        X_normalized = (X_test - first_data_point) / first_data_point
        X_test[:, :, :7] = X_normalized[:, :, :7]

    # test the model
    predictions = model.predict(X_test)

    date = i+1

    actual_change = y_test.item()
    predictions = predictions.item()

    print(f"predictions: {predictions}")

    #load into table
    cur.execute("INSERT INTO stock_predictions_v4 (date, actual_change, predicted_change) VALUES (%s, %s, %s)",
        (date, actual_change, predictions))


conn.commit()

conn.close()
