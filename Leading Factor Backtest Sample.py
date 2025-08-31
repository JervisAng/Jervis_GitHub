import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from scipy.stats import kurtosis


#===读取 Candle 数据
df_candle = pd.read_csv("data.csv")
df_candle['start_time'] = pd.to_datetime(df_candle['start_time'], unit='ms')

#===读取 Data 数据
df_premium = pd.read_csv("data.csv")
df_candle['data'] = df_premium['data']


df_candle = df_candle.dropna(subset=['data'])

df_candle = df_candle[['start_time', 'close', 'data']]

df_head = df_candle.head()
df_tail = df_candle.tail()

gap_row = pd.DataFrame([["..."] * df_candle.shape[1]], columns=df_candle.columns)

df_preview = pd.concat([df_head, gap_row, df_tail], ignore_index=True)

df_candle = df_candle[['start_time', 'close', 'coinbase_premium_index']]


#计算 change
df_candle['change'] = df_candle['close'].pct_change()


#计算 RW
RW = 100  
z_score = 100

df_candle['data'] = df_candle['data'].rolling(RW).mean()

df_candle.loc[:499, 'data'] = np.nan


#计算std
df_candle['std'] = df_candle['data'].rolling(RW).std()

df_candle.loc[:499, 'std'] = np.nan


#计算z-score
df_candle['z_score'] = (df_candle['data'] - df_candle['RW']) / df_candle['std']

df_candle.loc[:499, 'z_score'] = np.nan


#Position
start_idx = 497

df_candle.reset_index(drop=True, inplace=True)

df_candle['position'] = 0

for i in range(497, len(df_candle)):
    z = df_candle.loc[i, 'z_score']
    prev_pos = df_candle.loc[i - 1, 'position']
    
    if z <= -z_score:
        df_candle.loc[i, 'position'] = -1.
    elif z >= z_score:
        df_candle.loc[i, 'position'] = 1.
    else:
        df_candle.loc[i, 'position'] = prev_pos


#计算trade
start_idx = 497

df_candle.loc[:start_idx-1, 'position'] = 0

df_candle['trade'] = df_candle['position'].diff().abs()

df_candle.loc[:start_idx, 'trade'] = 0

df_candle['trade'] = df_candle['trade'].astype(int)


#计算Daily PnL
start_idx = 497
fee = 0.0006          

df_candle['prev_pos'] = df_candle['position'].shift(1).fillna(0)

df_candle.loc[:start_idx, 'prev_pos'] = 0

df_candle['Daily_PnL'] = df_candle['prev_pos'] * df_candle['change'] - df_candle['trade'] * fee

df_candle.loc[:497, 'Daily_PnL'] = np.nan


#计算Cumu (Equity Curve)
start_idx = 497
df_candle['cumu'] = df_candle['Daily_PnL'].cumsum()
df_candle.loc[:497, 'cumu'] = np.nan


# 计算 DrawDown
start_idx = 497
df_candle['DrawDown'] = df_candle['cumu'] - df_candle['cumu'].cummax()
df_candle.loc[:497, 'DrawDown'] = np.nan


#计算 Brought Down
start_idx = 498

brought_down = [np.nan] * len(df_candle)

brought_down[start_idx] = df_candle.loc[start_idx, 'cumu']

for i in range(start_idx + 1, len(df_candle)):
    if df_candle.loc[i, 'trade'] != 0:
        brought_down[i] = df_candle.loc[i, 'cumu']
    else:
        brought_down[i] = brought_down[i - 1]

df_candle['BroughtDown'] = brought_down


#取差值,记录 cumu
start_idx = 498

df_candle['取差值,记录cumu'] = df_candle['BroughtDown'] - df_candle['BroughtDown'].shift(1)
df_candle.loc[:start_idx, '取差值,记录cumu'] = 0 

df_candle.loc[:498, '取差值,记录cumu'] = np.nan


#计算 Winning Streak
start_idx = 497

winning_streak = [0] * len(df_candle)

for i in range(start_idx + 1, len(df_candle)):
    value = df_candle.loc[i, '取差值,记录cumu']
    
    if value > 0:
        winning_streak[i] = winning_streak[i - 1] + 1
    elif value < 0:
        winning_streak[i] = 0
    else:
        winning_streak[i] = winning_streak[i - 1]

df_candle['WinningStreak'] = winning_streak
df_candle.loc[:497, 'WinningStreak'] = np.nan


#计算 Losing Streak
start_idx = 497

losing_streak = [0] * len(df_candle)

for i in range(start_idx + 1, len(df_candle)):
    value = df_candle.loc[i, '取差值,记录cumu']

    if value < 0:
        losing_streak[i] = losing_streak[i - 1] + 1
    elif value > 0:
        losing_streak[i] = 0
    else:
        losing_streak[i] = losing_streak[i - 1]

df_candle['LosingStreak'] = losing_streak
df_candle.loc[:497, 'LosingStreak'] = np.nan


#计算 Recovery
start_idx = 497

for i in range(1, len(df_candle)):
    if df_candle.loc[i, 'DrawDown'] < 0:
        df_candle.loc[i, 'Recovery'] = df_candle.loc[i - 1, 'Recovery'] + 1
    else:
        df_candle.loc[i, 'Recovery'] = 0

df_candle.loc[:497, 'Recovery'] = np.nan



#计算Annual Return
Annual_Return = df_candle['Daily_PnL'].mean() * 365 * 24

#计算Sharpe Ratio
Sharpe_Ratio = df_candle['Daily_PnL'].mean()/df_candle['Daily_PnL'].std()*np.sqrt(365*24)

#计算MDD
MDD = df_candle['DrawDown'].min()

#j计算calmar ratio
Calmar_Ratio = Annual_Return / abs(MDD)

#计算Trade Count
Trade_Count = df_candle['trade'].sum()

#计算 Total Winning Streak
Winning_Streak = df_candle['WinningStreak'].max()

#计算 Total Losing Streak
Losing_Streak = df_candle['LosingStreak'].max()

#计算 Longest Recovery Day
Longest_Revovery_Day = df_candle['Recovery'].max()/24


#构建 preview（先 head，后 placeholder，再 tail）
df_preview = pd.concat([
    df_candle[['start_time', 'close', 'data']].head(),  #continue with Daily PnL, Cumu, Change ..........
    pd.DataFrame([['-', '-', '-']], columns=['start_time', 'close', 'data']),  #continue with Daily PnL, Cumu, Change ..........
    df_candle[['start_time', 'close', 'data']].tail()  #continue with Daily PnL, Cumu, Change ..........
])

print(df_preview)
print(Annual_Return)
print(Sharpe_Ratio)
print(MDD)
print(Calmar_Ratio)
print(Trade_Count)
print(Winning_Streak)
print(Losing_Streak)
print(Longest_Revovery_Day)

df_candle.to_csv("btc_1h_LeadingFactor_Backtest.csv", index=False)



