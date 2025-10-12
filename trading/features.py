import pandas as pd

y_horizon = 13

def calculate_vanilla_macd(series, slow=26, fast=12, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_hindsight_macd(series, slow=26, fast=12, signal=9):
    return calculate_vanilla_macd(series.shift(-y_horizon), fast, slow, signal)


def calculate_predicted_macd(df, predictions_df, short_window=12, long_window=26, signal_window=9):
    """
    Calculate the MACD using actual and predicted close prices for a given stock.

    Parameters:
    df (pd.DataFrame): DataFrame containing the actual close prices with a 'Close' column.
    predictions_df (pd.DataFrame): DataFrame containing predicted close prices for up to 13 days ahead.
    short_window (int): The short EMA window length (default is 12).
    long_window (int): The long EMA window length (default is 26).
    signal_window (int): The signal line EMA window length (default is 9).

    Returns:
    pd.DataFrame: Original DataFrame with added MACD and Signal columns.
    """
    results = {
        'Date': [],
        'Predicted_MACD': [],
        'Predicted_MACD_Signal_Line': []
    }

    for date in df.index:
        # Retrieve historical close prices up to the current date
        historical_prices = df.loc[:date, 'Close']

        # Retrieve future predictions for the current date
        if date in predictions_df.index:
            future_predictions = predictions_df.loc[date].values
        else:
            future_predictions = []

        # Combine historical closes with future predictions
        combined_series = pd.Series(list(historical_prices) + list(future_predictions))

        ema_fast = combined_series.ewm(span=short_window, adjust=False).mean()
        ema_slow = combined_series.ewm(span=long_window, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal_line = macd.ewm(span=signal_window, adjust=False).mean()

        results['Date'].append(date)
        results['Predicted_MACD'].append(macd.iloc[-1])
        results['Predicted_MACD_Signal_Line'].append(macd_signal_line.iloc[-1])

    results = pd.DataFrame(results).set_index('Date')

    return results

def calculate_volatility(series, window=5):
    return series.pct_change().rolling(window=window).std()

def calculate_roc(series, window=3):
    return series.pct_change(periods=window)

def calculate_rsi(df, period=14) -> pd.DataFrame:
    """
    Calculate the Relative Strength Index (RSI) for a given DataFrame of close prices.

    Parameters:
    df (pd.DataFrame): DataFrame containing the stock's close prices with a column 'Close'.
    period (int): The period over which to calculate the RSI (default is 14).

    Returns:
    pd.DataFrame: Original DataFrame with an additional column for RSI.
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_ideal_rsi(df, period=14) -> pd.DataFrame:
    return calculate_rsi(df.shift(-y_horizon), period)


def calculate_predicted_rsi(df, predictions_df, period=28):
    """
    Calculate the RSI (Relative Strength Index) using actual and predicted close prices for a given stock.

    Parameters:
    df (pd.DataFrame): DataFrame containing the actual close prices with a 'Close' column.
    predictions_df (pd.DataFrame): DataFrame containing predicted close prices for up to 13 days ahead.
    period (int): The period over which to calculate RSI (default is 14).

    Returns:
    pd.DataFrame: DataFrame containing the calculated RSI values based on actual and predicted data.
    """
    results = {
        'Date': [],
        'Predicted_RSI': []
    }

    for date in df.index:
        # Retrieve historical close prices up to the current date
        historical_prices = df.loc[:date, 'Close']

        # Retrieve future predictions for the current date
        if date in predictions_df.index:
            future_predictions = predictions_df.loc[date].values
        else:
            future_predictions = []

        # Combine historical closes with future predictions
        combined_series = pd.Series(list(historical_prices) + list(future_predictions))
        if len(combined_series) > period:
            combined_series = combined_series.iloc[-period:].reset_index(drop=True)

        # Calculate the changes in the combined series
        delta = combined_series.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate the average gains and losses over the specified period using rolling mean
        avg_gain = gains.rolling(window=period, min_periods=1).mean()
        avg_loss = losses.rolling(window=period, min_periods=1).mean()

        # Calculate the Relative Strength (RS) and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Store the results for the current date
        results['Date'].append(date)
        results['Predicted_RSI'].append(rsi.iloc[-1])

    # Convert results into a DataFrame
    results = pd.DataFrame(results).set_index('Date')

    return results


def calculate_aroon_oscillator(df, period=25) -> pd.DataFrame:
    """
    Calculate the Aroon Oscillator using the closing prices in a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the stock's close prices with a column 'Close'.
    period (int): The period over which to calculate the Aroon Oscillator (default is 25).

    Returns:
    pd.DataFrame: Original DataFrame with additional columns for Aroon Up, Aroon Down, and Aroon Oscillator.
    """
    df = df.copy()

    # Calculate the number of periods since the highest close over the specified period
    df['Periods_Since_High_Close'] = df['Close'].rolling(window=period, min_periods=1).apply(
        lambda x: period - x.argmax() - 1)

    # Calculate the number of periods since the lowest close over the specified period
    df['Periods_Since_Low_Close'] = df['Close'].rolling(window=period, min_periods=1).apply(
        lambda x: period - x.argmin() - 1)

    # Calculate Aroon Up: 100 * (period - Periods_Since_High_Close) / period
    df['Aroon_Up'] = 100 * (period - df['Periods_Since_High_Close']) / period

    # Calculate Aroon Down: 100 * (period - Periods_Since_Low_Close]) / period
    df['Aroon_Down'] = 100 * (period - df['Periods_Since_Low_Close']) / period

    # Calculate the Aroon Oscillator: Aroon Up - Aroon Down
    df['Aroon_Oscillator'] = df['Aroon_Up'] - df['Aroon_Down']

    return df[['Aroon_Oscillator']]


def calculate_ideal_aroon_oscillator(df, period=25) -> pd.DataFrame:
    ret = calculate_aroon_oscillator(df.shift(-y_horizon), period)
    return ret.rename(columns={'Aroon_Oscillator': 'Ideal_Aroon_Oscillator'})


def calculate_predicted_aroon(df, predictions_df, period=25):
    """
    Calculate the Aroon Oscillator using actual and predicted close prices for a given stock.

    Parameters:
    df (pd.DataFrame): DataFrame containing the actual close prices with a 'Close' column.
    predictions_df (pd.DataFrame): DataFrame containing predicted close prices for up to 13 days ahead.
    period (int): The lookback period for calculating the Aroon oscillator (default is 25).

    Returns:
    pd.DataFrame: DataFrame containing the calculated Aroon Up, Aroon Down, and Aroon Oscillator values based on actual and predicted data.
    """
    results = {
        'Date': [],
        'Predicted_Aroon_Up': [],
        'Predicted_Aroon_Down': [],
        'Predicted_Aroon_Oscillator': []
    }

    for date in df.index:
        # Retrieve historical close prices up to the current date
        historical_prices = df.loc[:date, 'Close']

        # Retrieve future predictions for the current date
        if date in predictions_df.index:
            future_predictions = predictions_df.loc[date].values
        else:
            future_predictions = []

        # Combine historical closes with future predictions
        combined_series = pd.Series(list(historical_prices) + list(future_predictions))
        if len(combined_series) > period:
            combined_series = combined_series.iloc[-period:].reset_index(drop=True)

        # Get the period window size
        window = min(period, len(combined_series))

        # Calculate the number of periods since the highest high and lowest low
        highest_high_idx = combined_series[-window:].idxmax()
        lowest_low_idx = combined_series[-window:].idxmin()

        periods_since_high = len(combined_series) - highest_high_idx - 1
        periods_since_low = len(combined_series) - lowest_low_idx - 1

        # Calculate Aroon Up and Aroon Down
        aroon_up = 100 * (period - periods_since_high) / period
        aroon_down = 100 * (period - periods_since_low) / period

        # Calculate the Aroon Oscillator
        aroon_oscillator = aroon_up - aroon_down

        # Store the results
        results['Date'].append(date)
        results['Predicted_Aroon_Up'].append(aroon_up)
        results['Predicted_Aroon_Down'].append(aroon_down)
        results['Predicted_Aroon_Oscillator'].append(aroon_oscillator)

    # Convert results into a DataFrame
    results = pd.DataFrame(results).set_index('Date')

    return results[['Predicted_Aroon_Oscillator']]


def calculate_stochastic_oscillator(df, period=14):
    """
    Calculate the Stochastic Oscillator (%K and %D) for a given DataFrame of high, low, and close prices.

    Parameters:
    df (pd.DataFrame): DataFrame containing the stock's high, low, and close prices with columns 'High', 'Low', and 'Close'.
    period (int): The period over which to calculate %K (default is 14).
    smooth_k (int): The smoothing period for %K (default is 3).
    smooth_d (int): The period for %D, which is a moving average of %K (default is 3).

    Returns:
    pd.DataFrame: Original DataFrame with additional columns for %K and %D.
    """
    df = df.copy()

    # Calculate the rolling highest high and lowest low over the specified period
    df['Highest_High'] = df['Close'].rolling(window=period, min_periods=1).max()
    df['Lowest_Low'] = df['Close'].rolling(window=period, min_periods=1).min()

    # Calculate %K (Fast Stochastic)
    df['Stochastic_Oscillator'] = 100 * (df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low'])

    return df[['Stochastic_Oscillator']]


def calculate_ideal_stochastic_oscillator(df, period=14):
    ret = calculate_stochastic_oscillator(df.shift(-y_horizon), period)
    return ret.rename(columns={'Stochastic_Oscillator': 'Ideal_Stochastic_Oscillator'})


def calculate_predicted_stochastic(df, predictions_df, period=14):
    """
    Calculate the Stochastic Oscillator (%K and %D) using actual and predicted close prices for a given stock.

    Parameters:
    df (pd.DataFrame): DataFrame containing the actual close prices with a 'Close' column.
    predictions_df (pd.DataFrame): DataFrame containing predicted close prices for up to 13 days ahead.
    k_period (int): The lookback period for calculating the %K line (default is 14).
    d_period (int): The period for calculating the %D (smoothed %K) line (default is 3).

    Returns:
    pd.DataFrame: DataFrame containing the calculated %K and %D values based on actual and predicted data.
    """
    results = {
        'Date': [],
        'Predicted_Stochastic_Oscillator': [],
    }

    for date in df.index:
        # Retrieve historical close prices up to the current date
        historical_prices = df.loc[:date, 'Close']

        # Retrieve future predictions for the current date
        if date in predictions_df.index:
            future_predictions = predictions_df.loc[date].values
        else:
            future_predictions = []

        # Combine historical closes with future predictions
        combined_series = pd.Series(list(historical_prices) + list(future_predictions))

        combined_series = combined_series
        if len(combined_series) > period:
            combined_series = combined_series.iloc[-period:].reset_index(drop=True)

        # Get the period window size
        window = min(period, len(combined_series))

        # Calculate the highest high and lowest low over the lookback period
        highest_high = combined_series[-window:].max()
        lowest_low = combined_series[-window:].min()

        # Calculate the current close (last value in the combined series)
        current_close = combined_series.iloc[-1]

        # Calculate %K
        if highest_high == lowest_low:
            percent_k = 0  # Avoid division by zero
        else:
            percent_k = 100 * (current_close - lowest_low) / (highest_high - lowest_low)

        # Store the results
        results['Date'].append(date)
        results['Predicted_Stochastic_Oscillator'].append(percent_k)

    # Convert results into a DataFrame
    results = pd.DataFrame(results).set_index('Date')

    return results

def calculate_bollinger_bands(series, window=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def calculate_bollinger_width(series, window=20, num_std_dev=2):
    """Calculates the width of the Bollinger Bands."""
    upper_band, lower_band = calculate_bollinger_bands(series, window, num_std_dev)
    return ((upper_band - lower_band) / series.rolling(window=window).mean()) * 100 # Normalized