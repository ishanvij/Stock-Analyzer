from flask import Flask, render_template, request, jsonify
import datetime
from utils import (fetch_data, analyze_and_forecast, make_interactive_plot, 
                  get_stock_info, calculate_technical_indicators)
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365*2)
    return render_template('index.html', default_ticker='AAPL', start=default_start.isoformat(), end=today.isoformat())

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.form
        symbol = data.get('symbol', 'AAPL').upper().strip()
        start = data.get('start')
        end = data.get('end')
        freq = data.get('freq', '1d')
        forecast_periods = int(data.get('periods', 30))
        analysis_type = data.get('analysis_type', 'basic')

        # Fetch stock data
        df = fetch_data(symbol, start, end, interval=freq)
        if df is None or df.empty:
            return jsonify({'status': 'error', 'message': f'No data available for {symbol}'}), 404

        # Add technical indicators if requested
        include_indicators = analysis_type in ['technical', 'complete']
        if include_indicators:
            df = calculate_technical_indicators(df)

        # Generate forecast
        fcst_result = analyze_and_forecast(df, periods=forecast_periods)
        
        # Create plot
        price_plot = make_interactive_plot(df, fcst_result, include_indicators=include_indicators)

        # Get additional stock information
        stock_info = get_stock_info(symbol)

        # Calculate daily change
        if len(df) > 1:
            current_price = df['Close'].iloc[-1]
            previous_price = df['Close'].iloc[-2]
            daily_change = current_price - previous_price
            daily_change_pct = (daily_change / previous_price) * 100
            daily_change_str = f"${daily_change:.2f} ({daily_change_pct:+.2f}%)"
        else:
            daily_change_str = '--'

        # Handle forecast table conversion
        forecast_table = []
        if 'forecast_table' in fcst_result and fcst_result['forecast_table'] is not None:
            try:
                if hasattr(fcst_result['forecast_table'], 'values'):
                    forecast_table = fcst_result['forecast_table'].values.tolist()
                else:
                    forecast_table = fcst_result['forecast_table']
            except:
                forecast_table = []

        response = {
            'status': 'ok',
            'symbol': symbol,
            'plot_html': price_plot,
            'model_summary': fcst_result.get('model_summary', ''),
            'forecast_table': forecast_table,
            'current_price': f"${stock_info.get('current_price', '--')}" if stock_info.get('current_price') != '--' else '--',
            'daily_change': daily_change_str,
            'volume': stock_info.get('volume', '--'),
            'market_cap': stock_info.get('market_cap', '--'),
            'pe_ratio': stock_info.get('pe_ratio', '--'),
            'dividend_yield': stock_info.get('dividend_yield', '--'),
            'company_name': stock_info.get('company_name', symbol),
            'sector': stock_info.get('sector', '--'),
            'industry': stock_info.get('industry', '--')
        }
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Analysis failed: {str(e)}'}), 500

@app.route('/compare', methods=['POST'])
def compare_stocks():
    try:
        data = request.form
        symbols = [s.strip().upper() for s in data.get('symbols', '').split(',') if s.strip()]
        period = data.get('period', '1y')
        
        if not symbols:
            return jsonify({'status': 'error', 'message': 'Please provide at least one stock symbol'}), 400
        
        if len(symbols) > 5:
            return jsonify({'status': 'error', 'message': 'Maximum 5 stocks can be compared at once'}), 400
        
        # Calculate date range
        end_date = datetime.date.today()
        if period == '1y':
            start_date = end_date - datetime.timedelta(days=365)
        elif period == '2y':
            start_date = end_date - datetime.timedelta(days=730)
        elif period == '6mo':
            start_date = end_date - datetime.timedelta(days=180)
        else:
            start_date = end_date - datetime.timedelta(days=365)
        
        comparison_data = []
        
        for symbol in symbols:
            try:
                df = fetch_data(symbol, start_date.isoformat(), end_date.isoformat())
                if df is not None and not df.empty:
                    stock_info = get_stock_info(symbol)
                    
                    # Calculate performance metrics
                    if len(df) > 1:
                        start_price = df['Close'].iloc[0]
                        end_price = df['Close'].iloc[-1]
                        total_return = ((end_price - start_price) / start_price) * 100
                        volatility = df['Close'].pct_change().std() * 100
                    else:
                        total_return = 0
                        volatility = 0
                    
                    comparison_data.append({
                        'symbol': symbol,
                        'current_price': stock_info.get('current_price', '--'),
                        'market_cap': stock_info.get('market_cap', '--'),
                        'pe_ratio': stock_info.get('pe_ratio', '--'),
                        'total_return': f"{total_return:.2f}%",
                        'volatility': f"{volatility:.2f}%",
                        'volume': stock_info.get('volume', '--')
                    })
                else:
                    comparison_data.append({
                        'symbol': symbol,
                        'error': f'No data available for {symbol}'
                    })
            except Exception as e:
                comparison_data.append({
                    'symbol': symbol,
                    'error': f'Error fetching data for {symbol}: {str(e)}'
                })
        
        return jsonify({
            'status': 'ok',
            'comparison_data': comparison_data,
            'period': period
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/market-indices', methods=['GET'])
def get_market_indices():
    """Get major market indices data"""
    try:
        indices = {
            'SP500': '^GSPC',
            'NASDAQ': '^IXIC', 
            'DOW': '^DJI'
        }
        
        market_data = {}
        
        for name, symbol in indices.items():
            try:
                # Get last 2 days of data to calculate change
                end_date = datetime.date.today()
                start_date = end_date - datetime.timedelta(days=5)
                
                df = fetch_data(symbol, start_date.isoformat(), end_date.isoformat())
                if df is not None and not df.empty and len(df) > 1:
                    current_price = df['Close'].iloc[-1]
                    previous_price = df['Close'].iloc[-2]
                    change = current_price - previous_price
                    change_pct = (change / previous_price) * 100
                    
                    market_data[name] = {
                        'price': f"{current_price:.2f}",
                        'change': f"{change:+.2f}",
                        'change_pct': f"{change_pct:+.2f}%"
                    }
                else:
                    market_data[name] = {'error': 'No data available'}
            except Exception as e:
                market_data[name] = {'error': str(e)}
        
        return jsonify({'status': 'ok', 'data': market_data})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
