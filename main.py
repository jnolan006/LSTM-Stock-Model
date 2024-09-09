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

# np.set_printoptions(threshold=np.inf)

# Connect to database
host = "ec2-52-6-117-96.compute-1.amazonaws.com"
dbname = "dftej5l5m1cl78"
user = "aiuhlrpcnftsjs"
password = "8b2220cd5b6da572369545d91f6b435dfc37a42bfec6b6e2a5c9f236dfb65f42"

conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
cur = conn.cursor()

# load historical stock data
# query = "SELECT date, close, LN(close / LAG(close) OVER (ORDER BY date)) AS ln_close, LN(open / LAG(open) OVER (ORDER BY date)) AS open, LN(ema_200 / LAG(ema_200) OVER (ORDER BY date)) AS ema_200, LN(ema_50 / LAG(ema_50) OVER (ORDER BY date)) AS ema_50, LN(ema_12 / LAG(ema_12) OVER (ORDER BY date)) AS ema_12, LN(ema_26 / LAG(ema_26) OVER (ORDER BY date)) AS ema_26, LN(upper_band / LAG(upper_band) OVER (ORDER BY date)) AS upper_band, LN(lower_band / LAG(lower_band) OVER (ORDER BY date)) AS lower_band, macd_results, rsi, stochastic_oscillator, LN(high / LAG(high) OVER (ORDER BY date)) AS high, LN(low / LAG(low) OVER (ORDER BY date)) AS low, volume, LN(ema_1 / LAG(ema_1) OVER (ORDER BY date)) AS ema_1 FROM daily_stock_price ORDER BY date;"
# query = "SELECT date, CASE WHEN (LEAD(close, 1) OVER (ORDER BY date) > LEAD(open, 1) OVER (ORDER BY date)) AND (LEAD(close, 2) OVER (ORDER BY date) > LEAD(open, 2) OVER (ORDER BY date)) AND (LEAD(close, 3) OVER (ORDER BY date) > LEAD(open, 3) OVER (ORDER BY date)) THEN 1 WHEN (LEAD(close, 1) OVER (ORDER BY date) < LEAD(open, 1) OVER (ORDER BY date)) AND (LEAD(close, 2) OVER (ORDER BY date) < LEAD(open, 2) OVER (ORDER BY date)) AND (LEAD(close, 3) OVER (ORDER BY date) < LEAD(open, 3) OVER (ORDER BY date)) THEN -1 ELSE 0 END as category, close, LN(close / LAG(close) OVER (ORDER BY date)) AS ln_close, open, ema_200, ema_50, ema_12, ema_26, upper_band, lower_band, macd_results, rsi, stochastic_oscillator, high, low, volume, ema_1 FROM daily_stock_price ORDER BY date;"
# query = "SELECT date, CASE WHEN close-open > 0 then 1 when close-open <= 0 then 0 END as category, close, open, ema_200, ema_50, ema_12, ema_26, high, low, volume, ((close - open) / open) * 100 AS daily_percent_change, ((open - LAG(close) OVER (ORDER BY symbol, date)) / LAG(close) OVER (ORDER BY symbol, date)) * 100 AS overnight_percent_change, ((high - low) / low) * 100 AS percentage_volatility, ABS((close - open) / open) * 100 AS absolute_percent_change, EXTRACT(DOW FROM date) AS day_of_week, case when grp = 1 and lag(grp) over(partition by symbol order by date) = 0 then 1 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 0 then 2 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 0 then 3 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 0 then 4 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 0 then 5 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 1 and lag(grp,6) over(partition by symbol order by date) = 0 then 6 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 1 and lag(grp,6) over(partition by symbol order by date) = 1 and lag(grp,7) over(partition by symbol order by date) = 0 then 7 else 0 end as consecutive_green_days, case when grp = 0 and lag(grp) over(partition by symbol order by date) = 1 then 1 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 1 then 2 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 1 then 3 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 1 then 4 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 1 then 5 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 1 then 6 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 0 and lag(grp,7) over(partition by symbol order by date) = 1 then 7 else 0 end as consecutive_red_days, symbol FROM (SELECT *, CASE WHEN close > open THEN 1 when close <= open then 0 END AS grp FROM daily_stock_price) as test ORDER BY symbol, date;"
# query = "SELECT date, CASE WHEN close-open > 0 then 1 when close-open <= 0 then 0 END as category, volume, rsi, ((close - open) / open) * 100 AS daily_percent_change, ((open - LAG(close) OVER (ORDER BY symbol, date)) / LAG(close) OVER (ORDER BY symbol, date)) * 100 AS overnight_percent_change, ((high - low) / low) * 100 AS percentage_volatility, ABS((close - open) / open) * 100 AS absolute_percent_change, EXTRACT(DOW FROM date) AS day_of_week, case when grp = 1 and lag(grp) over(partition by symbol order by date) = 0 then 1 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 0 then 2 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 0 then 3 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 0 then 4 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 0 then 5 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 1 and lag(grp,6) over(partition by symbol order by date) = 0 then 6 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 1 and lag(grp,6) over(partition by symbol order by date) = 1 and lag(grp,7) over(partition by symbol order by date) = 0 then 7 else 0 end as consecutive_green_days, case when grp = 0 and lag(grp) over(partition by symbol order by date) = 1 then 1 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 1 then 2 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 1 then 3 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 1 then 4 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 1 then 5 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 1 then 6 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 0 and lag(grp,7) over(partition by symbol order by date) = 1 then 7 else 0 end as consecutive_red_days,     volume / AVG(volume) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) AS relative_volume, symbol FROM (SELECT *, CASE WHEN close > open THEN 1 when close <= open then 0 END AS grp FROM daily_stock_price) as test ORDER BY symbol, date;"
# query = "SELECT date, CASE WHEN close-open > 0 then 1 when close-open <= 0 then 0 END as category, open, symbol FROM (SELECT *, CASE WHEN close > open THEN 1 when close <= open then 0 END AS grp FROM daily_stock_price) as test ORDER BY symbol, date;"
query = "SELECT date, symbol, CASE WHEN close-open < 0 then 1 when close-open >= 0 then 0 END as category, close, open, ema_200, ema_50, ema_12, ema_26, high, low, volume, upper_band, lower_band, macd_results, rsi, stochastic_oscillator, ((close - open) / open) * 100 AS daily_percent_change, ((open - LAG(close) OVER (ORDER BY symbol, date)) / LAG(close) OVER (ORDER BY symbol, date)) * 100 AS overnight_percent_change, ((high - low) / low) * 100 AS percentage_volatility, ABS((close - open) / open) * 100 AS absolute_percent_change, EXTRACT(DOW FROM date) AS day_of_week, case when grp = 1 and lag(grp) over(partition by symbol order by date) = 0 then 1 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 0 then 2 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 0 then 3 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 0 then 4 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 0 then 5 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 1 and lag(grp,6) over(partition by symbol order by date) = 0 then 6 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 1 and lag(grp,6) over(partition by symbol order by date) = 1 and lag(grp,7) over(partition by symbol order by date) = 0 then 7 else 0 end as consecutive_green_days, case when grp = 0 and lag(grp) over(partition by symbol order by date) = 1 then 1 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 1 then 2 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 1 then 3 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 1 then 4 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 1 then 5 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 1 then 6 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 0 and lag(grp,7) over(partition by symbol order by date) = 1 then 7 else 0 end as consecutive_red_days,     volume / AVG(volume) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) AS relative_volume FROM (SELECT *, CASE WHEN close > open THEN 1 when close <= open then 0 END AS grp FROM daily_stock_price_fin) as test ORDER BY symbol, date;"
# query = "select date, symbol, CASE WHEN (close-open)/open*100 > 1 or (close-open)/open*100 < -1 then 1 else 0 end as category, close, open, ema_200, ema_12, high, low, volume, lower_band, macd_results, rsi, stochastic_oscillator, ((open - LAG(close) OVER (ORDER BY symbol, date)) / LAG(close) OVER (ORDER BY symbol, date)) * 100 AS overnight_percent_change, ((high - low) / low) * 100 AS percentage_volatility, case when grp = 0 and lag(grp) over(partition by symbol order by date) = 1 then 1 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 1 then 2 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 1 then 3 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 1 then 4 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 1 then 5 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 1 then 6 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 0 and lag(grp,7) over(partition by symbol order by date) = 1 then 7 else 0 end as consecutive_red_days,     volume / AVG(volume) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) AS relative_volume FROM (SELECT *, CASE WHEN close > open THEN 1 when close <= open then 0 END AS grp FROM daily_stock_price) as test ORDER BY symbol, date;"
# query = "SELECT date, CASE WHEN close-open > 0 then 1 when close-open <= 0 then 0 END as category,  FROM (SELECT *, CASE WHEN close > open THEN 1 when close <= open then 0 END AS grp FROM daily_stock_price) as test ORDER BY symbol, date;"
# query = "SELECT date, CASE WHEN (close-open)/open*100 > 1 or (close-open)/open*100 < -1 then 1 else 0 end as category, close, open, ema_200, ema_50, ema_12, ema_26, high, low, volume, ((close - open) / open) * 100 AS daily_percent_change, ((open - LAG(close) OVER (ORDER BY symbol, date)) / LAG(close) OVER (ORDER BY symbol, date)) * 100 AS overnight_percent_change, ((high - low) / low) * 100 AS percentage_volatility, ABS((close - open) / open) * 100 AS absolute_percent_change, EXTRACT(DOW FROM date) AS day_of_week, case when grp = 1 and lag(grp) over(partition by symbol order by date) = 0 then 1 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 0 then 2 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 0 then 3 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 0 then 4 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 0 then 5 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 1 and lag(grp,6) over(partition by symbol order by date) = 0 then 6 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 1 and lag(grp,6) over(partition by symbol order by date) = 1 and lag(grp,7) over(partition by symbol order by date) = 0 then 7 else 0 end as consecutive_green_days, case when grp = 0 and lag(grp) over(partition by symbol order by date) = 1 then 1 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 1 then 2 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 1 then 3 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 1 then 4 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 1 then 5 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 1 then 6 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 0 and lag(grp,7) over(partition by symbol order by date) = 1 then 7 else 0 end as consecutive_red_days, symbol FROM (SELECT *, CASE WHEN close > open THEN 1 when close <= open then 0 END AS grp FROM daily_stock_price) as test ORDER BY symbol, date;"
cur.execute(query)
data = cur.fetchall()

# extract the data into a DataFrame
# data = pd.DataFrame(data, columns=['date', 'category', 'close', 'ln_close', 'open', 'ema_200', 'ema_50', 'ema_12', 'ema_26', 'upper_band', 'lower_band', 'macd_results', 'rsi', 'stochastic_oscillator', 'high', 'low', 'volume', 'ema_1'])
# data = pd.DataFrame(data, columns=['date', 'category', 'volume', 'rsi', 'daily_percent_change', 'overnight_percent_change', 'percentage_volatility', 'absolute_percent_change', 'day_of_week', 'consecutive_green_days',  'consecutive_red_days', 'relative_volume', 'symbol'])
# data = pd.DataFrame(data, columns=['date', 'category', 'open', 'symbol'])
# data = pd.DataFrame(data, columns=['date', 'category', 'volume', 'overnight_percent_change', 'percentage_volatility', 'relative_volume', 'symbol'])
data = pd.DataFrame(data, columns=[ "date", "symbol", "category", "close", "open", "ema_200", "ema_50", "ema_12", "ema_26", "high", "low", "volume", "upper_band", "lower_band", "macd_results", "rsi", "stochastic_oscillator", "daily_percent_change", "overnight_percent_change", "percentage_volatility", "absolute_percent_change", "day_of_week", "consecutive_green_days",  "consecutive_red_days", "relative_volume"])
# data = pd.DataFrame(data, columns=[ "date", "symbol", "category", "close", "open", "ema_200", "ema_12", "high", "low", "volume", "lower_band", "macd_results", "rsi", "stochastic_oscillator", "overnight_percent_change", "percentage_volatility", "consecutive_red_days", "relative_volume"])

# Sort data by date in ascending order
data = data.sort_values(by=['symbol', 'date'], ascending=[True, True])

data = data.iloc[:-1]

scaled_data_2 = data.iloc[100:]

sc = MinMaxScaler()

# Define your date range
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

# Get unique stock symbols
symbols = data['symbol'].unique()
# Create empty lists to store features (X) and target values (y)
X_train = []
y_train = []

for symbol in symbols:
    # Filter data for the current symbol
    symbol_data = scaled_data_2[scaled_data_2['symbol'] == symbol]

    # Find the indices of the rows that match the date range
    date_range_data = symbol_data[(symbol_data['date'] >= start_date) & (symbol_data['date'] <= end_date)]

    # Assuming that 'category' is the column representing the target variable
    target_variable = 'category'

    # Number of data points in each group
    data_points_in_group = 20

    # Iterate through the data with a sliding window
    for i in range(len(date_range_data) - data_points_in_group - 1):
        # Extract the features for the current group
        current_group_features = date_range_data.iloc[i:i + data_points_in_group, 2:].values
        sc.fit(current_group_features)
        current_group_features_scaled = sc.transform(current_group_features)

        # Extract the target value for the next day
        target_value = date_range_data.iloc[i + data_points_in_group + 1, 2]

        # Append the features and target value to the respective lists
        X_train.append(current_group_features_scaled)
        y_train.append(target_value)


# Convert lists to NumPy arrays
X_train_ema_2 = np.array(X_train)
Y_train_ema_2 = np.array(y_train)

# Convert the overall lists to NumPy arrays
X_train_ema_2, Y_train_ema_2 = np.array(X_train_ema_2), np.array(Y_train_ema_2)

# # Find the indices where Y value is 0
# zero_indices = np.where(Y_train_ema_2 == 0)[0]

# # Set the number of records to drop
# records_to_drop = 0

# # Randomly select a subset of indices with Y value 0 to drop
# indices_to_drop = np.random.choice(zero_indices, size=min(records_to_drop, len(zero_indices)), replace=False)

# # Drop the selected instances from X_train_ema_2 and Y_train_ema_2
# X_train_ema_2_balanced = np.delete(X_train_ema_2, indices_to_drop, axis=0)
# Y_train_ema_2_balanced = np.delete(Y_train_ema_2, indices_to_drop)

# X_train_ema_2_balanced_2, Y_train_ema_2_balanced_2 = np.array(X_train_ema_2_balanced), np.array(Y_train_ema_2_balanced)


# Print the counts after balancing
# unique_values_balanced, counts_balanced = np.unique(Y_train_ema_2_balanced, return_counts=True)
# for value, count in zip(unique_values_balanced, counts_balanced):
#     print(f"Y value {value}: {count} occurrences (after dropping 3000 records)")

# Reshape the input data
X_train_ema_3 = X_train_ema_2.reshape((X_train_ema_2.shape[0], -1))
print("X_train_ema_3 shape:", X_train_ema_3.shape)

X_train, X_val, y_train, y_val = train_test_split(X_train_ema_3, Y_train_ema_2, test_size=0.3, random_state=42)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)

clf = xgb.XGBClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model as a pickle (pkl) file
with open('xgboost_model_neg_fin.pkl', 'wb') as pkl_file:
    pickle.dump(clf, pkl_file)

# # Make predictions
# predictions = clf.predict(X_val)

# Make predictions
raw_predictions = clf.predict_proba(X_val)[:, 1]  # Get the raw probability estimates for class 1

# Set a custom threshold (you can adjust this)
custom_threshold = 0.7

# Convert raw probabilities to binary predictions using the threshold
binary_predictions = (raw_predictions >= custom_threshold).astype(int)

# clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf_rf.fit(X_train, y_train)

# # Save the Random Forest model as a pickle (pkl) file
# with open('random_forest_model.pkl', 'wb') as pkl_file:
#     pickle.dump(clf_rf, pkl_file)

# # Make predictions
# predictions = clf_rf.predict(X_val)

# num_plots = len(predictions) // 50

# # Assuming y_pred_filtered contains your predictions
# nan_indices = np.isnan(predictions)

# # Remove rows with NaN predictions
# X_test_filtered_no_nan = X_val[~nan_indices]
# y_test_filtered_no_nan = y_val[~nan_indices]
# y_pred_filtered_no_nan = predictions[~nan_indices]

# for i in range(num_plots):
#     start_idx = i * 50
#     end_idx = (i + 1) * 50

#     # Scatter plot for each set of 50 samples
#     plt.scatter(range(start_idx + 1, end_idx + 1), y_pred_filtered_no_nan[start_idx:end_idx], label='Predictions', marker='o')
#     plt.scatter(range(start_idx + 1, end_idx + 1), y_test_filtered_no_nan[start_idx:end_idx], label='Actual', marker='x')
#     plt.title(f'Decision Tree Predictions vs Actual (Samples {start_idx + 1}-{end_idx})')
#     plt.xlabel('Sample')
#     plt.ylabel('Category')
#     plt.legend()
#     plt.show()


# Print the counts for each category
category_0_count = np.sum(binary_predictions == 0)
category_1_count = np.sum(binary_predictions == 1)

print(f'Predicted Category 0 count: {category_0_count}')
print(f'Predicted Category 1 count: {category_1_count}')

# Calculate accuracy for each category (0 and 1)
total_category_0 = np.sum(binary_predictions == 0)
correct_category_0 = np.sum((y_val == 0) & (binary_predictions == 0))

total_category_1 = np.sum(binary_predictions == 1)
correct_category_1 = np.sum((y_val == 1) & (binary_predictions == 1))

# Calculate accuracy scores
category_0_accuracy = correct_category_0 / total_category_0
category_1_accuracy = correct_category_1 / total_category_1

# Print the results
print(f'Category 0 Accuracy: {category_0_accuracy}')
print(f'Category 1 Accuracy: {category_1_accuracy}')


# individual_trees = clf.estimators_

# # Visualize each tree in the Random Forest
# for i, tree in enumerate(individual_trees):
#     plt.figure(figsize=(20, 10))
#     plot_tree(tree, filled=True, feature_names=[f'feature_{i}' for i in range(X_train_ema_4.shape[1])], class_names=['0', '1'], rounded=True, fontsize=10)
#     plt.title(f'Decision Tree {i+1}')
#     plt.show()

column_names = scaled_data_2.columns.tolist()  # Assuming 'data' is your original dataset

# Close the database connection
conn.close()
