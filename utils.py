import yfinance as yf
import pandas as pd
import numpy as np
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objs as go
import plotly.io as pio
from joblib import Memory
import os

CACHE_DIR = os.path.join('.', 'cache')
memory = Memory(CACHE_DIR, verbose=0)

@memory.cache
def fetch_data(symbol: str, start: str, end: str, interval: str = '1d'):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
        if df.empty:
            return df
        df = df.reset_index()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        return df
    except Exception:
        return pd.DataFrame()

def get_stock_info(symbol: str):
    """Get additional stock information like market cap, P/E ratio, etc."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', '--')),
            'market_cap': format_number(info.get('marketCap', 0)),
            'pe_ratio': info.get('trailingPE', '--'),
            'dividend_yield': f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else '--',
            'volume': format_number(info.get('volume', 0)),
            'avg_volume': format_number(info.get('averageVolume', 0)),
            'company_name': info.get('longName', symbol),
            'sector': info.get('sector', '--'),
            'industry': info.get('industry', '--')
        }
    except Exception:
        return {}

def format_number(num):
    """Format large numbers with appropriate suffixes."""
    if num == 0 or num == '--':
        return '--'
    
    try:
        num = float(num)
        if num >= 1e12:
            return f"{num/1e12:.2f}T"
        elif num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return f"{num:.2f}"
    except:
        return '--'

def calculate_technical_indicators(df):
    """Calculate various technical indicators."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Moving Averages
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    df['RSI'] = calculate_rsi(df['Close'])
    
    # MACD
    macd_line, macd_signal, macd_histogram = calculate_macd(df['Close'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = macd_signal
    df['MACD_Histogram'] = macd_histogram
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    
    return df

def calculate_rsi(prices, window=14):
    """Calculate RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def fit_auto_arima(series, seasonal=False, m=1):
    model = auto_arima(series, seasonal=seasonal, m=m, stepwise=True, suppress_warnings=True, error_action='ignore', trace=False)
    return model

def fit_sarimax(series, order=(1,1,1), seasonal_order=(0,0,0,0)):
    mod = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False)
    return res

def analyze_and_forecast(df, periods=30):
    result = {}
    series = df['Close'].asfreq('D').ffill()
    
    # Try pmdarima first if available
    if PMDARIMA_AVAILABLE:
        try:
            arima_model = fit_auto_arima(series, seasonal=False)
            fcst, conf_int = arima_model.predict(n_periods=periods, return_conf_int=True, alpha=0.05)
            index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
            forecast_df = pd.DataFrame({'ds': index, 'yhat': fcst, 'yhat_lower': conf_int[:,0], 'yhat_upper': conf_int[:,1]}).set_index('ds')
            result['method'] = 'pmdarima_auto_arima'
            result['forecast'] = forecast_df
            result['model_summary'] = str(arima_model.summary())
            if not result['forecast'].empty:
                result['forecast_table'] = result['forecast'][['yhat','yhat_lower','yhat_upper']].round(2)
            return result
        except Exception as e:
            pass  # Fall through to SARIMAX
    
    # Use SARIMAX as fallback
    try:
        sarimax_res = fit_sarimax(series)
        pred = sarimax_res.get_forecast(steps=periods)
        mean = pred.predicted_mean
        ci = pred.conf_int()
        index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
        forecast_df = pd.DataFrame({'ds': index, 'yhat': mean.values, 'yhat_lower': ci.iloc[:,0].values, 'yhat_upper': ci.iloc[:,1].values}).set_index('ds')
        result['method'] = 'sarimax'
        result['forecast'] = forecast_df
        result['model_summary'] = str(sarimax_res.summary())
    except Exception as e2:
        result['method'] = 'none'
        result['forecast'] = pd.DataFrame()
        result['model_summary'] = f'Model fitting failed: {e2}'

    if not result['forecast'].empty:
        result['forecast_table'] = result['forecast'][['yhat','yhat_lower','yhat_upper']].round(2)
    return result

def make_interactive_plot(df, fcst_result, include_indicators=False):
    """Create interactive plot with optional technical indicators."""
    pio.templates.default = "plotly"
    
    # Main price chart
    hist = df['Close'].reset_index()
    
    data = []
    
    # Price line
    hist_trace = go.Scatter(
        x=hist['Date'], 
        y=hist['Close'], 
        mode='lines', 
        name='Close Price',
        line=dict(color='#667eea', width=2)
    )
    data.append(hist_trace)
    
    # Add technical indicators if requested
    if include_indicators and len(df) > 50:
        # Moving Averages
        if 'MA_20' in df.columns:
            ma20 = df['MA_20'].reset_index()
            ma20_trace = go.Scatter(
                x=ma20['Date'], 
                y=ma20['MA_20'], 
                mode='lines', 
                name='MA 20',
                line=dict(color='orange', width=1)
            )
            data.append(ma20_trace)
        
        if 'MA_50' in df.columns:
            ma50 = df['MA_50'].reset_index()
            ma50_trace = go.Scatter(
                x=ma50['Date'], 
                y=ma50['MA_50'], 
                mode='lines', 
                name='MA 50',
                line=dict(color='red', width=1)
            )
            data.append(ma50_trace)
        
        # Bollinger Bands
        if 'BB_Upper' in df.columns:
            bb_data = df[['BB_Upper', 'BB_Lower']].reset_index()
            bb_upper = go.Scatter(
                x=bb_data['Date'], 
                y=bb_data['BB_Upper'], 
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(128,128,128,0.5)'),
                showlegend=False
            )
            bb_lower = go.Scatter(
                x=bb_data['Date'], 
                y=bb_data['BB_Lower'], 
                mode='lines',
                name='Bollinger Bands',
                line=dict(color='rgba(128,128,128,0.5)'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            )
            data.extend([bb_upper, bb_lower])
    
    # Add forecast if available
    if 'forecast' in fcst_result and not fcst_result['forecast'].empty:
        fcst = fcst_result['forecast'].reset_index()
        trace_fcst = go.Scatter(
            x=fcst['ds'], 
            y=fcst['yhat'], 
            mode='lines', 
            name='Forecast',
            line=dict(color='green', width=2, dash='dash')
        )
        
        # Confidence interval
        ci = go.Scatter(
            x=list(fcst['ds']) + list(fcst['ds'][::-1]),
            y=list(fcst['yhat_upper']) + list(fcst['yhat_lower'][::-1]),
            fill='toself',
            fillcolor='rgba(0,128,0,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='95% Confidence Interval',
        )
        data.extend([trace_fcst, ci])
    
    # Create layout
    layout = go.Layout(
        title={
            'text': 'Stock Price Analysis & Forecast',
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis=dict(title='Date', showgrid=True),
        yaxis=dict(title='Price ($)', showgrid=True),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        height=500
    )
    
    fig = go.Figure(data=data, layout=layout)
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
