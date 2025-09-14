import numpy as np

y_horizon = 13

def target_log_return(df, t):
    return np.log(df['Close'].shift(-t) / df['Close'])

def log_return(df, days, delay = 0):
    return np.log(df['Close'].shift(delay) / df['Close'].shift(delay + days))

def over_night_return(df, delay=0):
    return np.log(df['Open'].shift(delay) / df['Close'].shift(delay + 1))

def day_range(df, delay=0):
    return np.log(df['High'].shift(delay) / df['Low'].shift(delay))

def momentum(df, days):
    return log_return(df, days, delay=0)

def intraday_log_return(df, delay=0):
    return np.log(df['Close'].shift(delay) / df['Open'].shift(delay))

def momentum_change(df, period1, period2):
    momentum1 = momentum(df, period1)
    momentum2 = momentum(df, period2)
    return momentum1 - momentum2

def up_days_count(df, days):
    return df['Close'].diff().rolling(window=days).apply(lambda x: (x > 0).sum())

def notional_traded(df):
    return df['Close'] * df['Volume']

def notional_traded_change(df, days, delay=0):
    notional = notional_traded(df)
    return notional.pct_change(days).shift(delay)

def coefficient_of_variation(df, days):
    return  df['Close'].rolling(days).std() / df['Close'].rolling(days).mean()

def returns_std(df, days):
    return log_return(df, 1,0).rolling(days).std()