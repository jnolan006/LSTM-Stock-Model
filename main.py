import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from datetime import date, timedelta
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
from sklearn.preprocessing import StandardScaler

# Connect to database and extract data
host = "ec2-52-6-117-96.compute-1.amazonaws.com"
dbname = "dftej5l5m1cl78"
user = "aiuhlrpcnftsjs"
password = "8b2220cd5b6da572369545d91f6b435dfc37a42bfec6b6e2a5c9f236dfb65f42"

conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
cur = conn.cursor()

query = "SELECT date, symbol, CASE WHEN close-open < 0 then 1 when close-open >= 0 then 0 END as category, close, open, ema_200, ema_50, ema_12, ema_26, high, low, volume, upper_band, lower_band, macd_results, rsi, stochastic_oscillator, ((close - open) / open) * 100 AS daily_percent_change, ((open - LAG(close) OVER (ORDER BY symbol, date)) / LAG(close) OVER (ORDER BY symbol, date)) * 100 AS overnight_percent_change, ((high - low) / low) * 100 AS percentage_volatility, ABS((close - open) / open) * 100 AS absolute_percent_change, EXTRACT(DOW FROM date) AS day_of_week, case when grp = 1 and lag(grp) over(partition by symbol order by date) = 0 then 1 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 0 then 2 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 0 then 3 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 0 then 4 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 0 then 5 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 1 and lag(grp,6) over(partition by symbol order by date) = 0 then 6 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 1 and lag(grp,6) over(partition by symbol order by date) = 1 and lag(grp,7) over(partition by symbol order by date) = 0 then 7 else 0 end as consecutive_green_days, case when grp = 0 and lag(grp) over(partition by symbol order by date) = 1 then 1 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 1 then 2 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 1 then 3 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 1 then 4 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 1 then 5 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 1 then 6 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 0 and lag(grp,7) over(partition by symbol order by date) = 1 then 7 else 0 end as consecutive_red_days,     volume / AVG(volume) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) AS relative_volume FROM (SELECT *, CASE WHEN close > open THEN 1 when close <= open then 0 END AS grp FROM daily_stock_price_fin) as test ORDER BY symbol, date;"
cur.execute(query)
data = cur.fetchall()

data = pd.DataFrame(data, columns=[ "date", "symbol", "category", "close", "open", "ema_200", "ema_50", "ema_12", "ema_26", "high", "low", "volume", "upper_band", "lower_band", "macd_results", "rsi", "stochastic_oscillator", "daily_percent_change", "overnight_percent_change", "percentage_volatility", "absolute_percent_change", "day_of_week", "consecutive_green_days",  "consecutive_red_days", "relative_volume"])

# Format data for model

data = data.sort_values(by=['symbol', 'date'], ascending=[True, True])

data = data.iloc[:-1]

scaled_data_2 = data.iloc[100:]

sc = MinMaxScaler()

start_date = date(2000, 1, 1)
end_date = date(2023, 1, 1)

if start_date not in scaled_data_2['date'].values:
    start_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    start_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    start_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    start_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    start_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    end_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    end_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    end_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    end_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    end_date += timedelta(days=1)

symbols = data['symbol'].unique()

X_train = []
y_train = []

for symbol in symbols:
    symbol_data = scaled_data_2[scaled_data_2['symbol'] == symbol]

    date_range_data = symbol_data[(symbol_data['date'] >= start_date) & (symbol_data['date'] <= end_date)]

    target_variable = 'category'

    data_points_in_group = 20

    for i in range(len(date_range_data) - data_points_in_group - 1):
        current_group_features = date_range_data.iloc[i:i + data_points_in_group, 2:].values
        sc.fit(current_group_features)
        current_group_features_scaled = sc.transform(current_group_features)

        target_value = date_range_data.iloc[i + data_points_in_group + 1, 2]

        X_train.append(current_group_features_scaled)
        y_train.append(target_value)


X_train_ema_2 = np.array(X_train)
Y_train_ema_2 = np.array(y_train)

X_train_ema_2, Y_train_ema_2 = np.array(X_train_ema_2), np.array(Y_train_ema_2)

X_train_ema_3 = X_train_ema_2.reshape((X_train_ema_2.shape[0], -1))
print("X_train_ema_3 shape:", X_train_ema_3.shape)

X_train, X_val, y_train, y_val = train_test_split(X_train_ema_3, Y_train_ema_2, test_size=0.3, random_state=42)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)

# Train model
clf = xgb.XGBClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model as a pickle (pkl) file
with open('xgboost_model_neg_fin.pkl', 'wb') as pkl_file:
    pickle.dump(clf, pkl_file)

# test trained model
raw_predictions = clf.predict_proba(X_val)[:, 1]  

custom_threshold = 0.7

binary_predictions = (raw_predictions >= custom_threshold).astype(int)

category_0_count = np.sum(binary_predictions == 0)
category_1_count = np.sum(binary_predictions == 1)

print(f'Predicted Category 0 count: {category_0_count}')
print(f'Predicted Category 1 count: {category_1_count}')

total_category_0 = np.sum(binary_predictions == 0)
correct_category_0 = np.sum((y_val == 0) & (binary_predictions == 0))

total_category_1 = np.sum(binary_predictions == 1)
correct_category_1 = np.sum((y_val == 1) & (binary_predictions == 1))

category_0_accuracy = correct_category_0 / total_category_0
category_1_accuracy = correct_category_1 / total_category_1

# Print the results
print(f'Category 0 Accuracy: {category_0_accuracy}')
print(f'Category 1 Accuracy: {category_1_accuracy}')

column_names = scaled_data_2.columns.tolist()  

conn.close()
