# Stock Market & Time Series Analyzer

## Overview

This is a Flask-based web application for stock market analysis and forecasting. The application provides real-time stock data visualization, technical analysis, and time series forecasting capabilities. Users can analyze individual stocks with interactive charts, technical indicators, and AI-powered price predictions using ARIMA models.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Web Framework**: Traditional server-rendered Flask application with HTML templates
- **UI Components**: Custom CSS with gradient styling and responsive design
- **Interactivity**: Vanilla JavaScript for tab navigation, form handling, and dynamic content updates
- **Visualization**: Plotly.js for interactive financial charts and time series plots
- **Styling**: Custom CSS with modern design patterns including backdrop filters and gradients

### Backend Architecture
- **Web Framework**: Flask with RESTful API endpoints
- **Data Processing**: Pandas and NumPy for financial data manipulation and analysis
- **Forecasting Engine**: 
  - Primary: pmdarima (auto-ARIMA) for automated time series forecasting
  - Fallback: statsmodels SARIMAX for manual ARIMA modeling
- **Caching Strategy**: joblib Memory for caching expensive API calls and computations
- **Error Handling**: Comprehensive try-catch blocks with graceful degradation

### Data Architecture
- **Data Source**: Yahoo Finance API via yfinance library for real-time and historical stock data
- **Data Storage**: In-memory processing with file-based caching (no persistent database)
- **Data Pipeline**: 
  1. Fetch raw stock data from Yahoo Finance
  2. Clean and normalize data using pandas
  3. Calculate technical indicators (moving averages, RSI, etc.)
  4. Apply time series forecasting models
  5. Generate interactive visualizations

### Technical Analysis Components
- **Price Analysis**: OHLCV (Open, High, Low, Close, Volume) data processing
- **Technical Indicators**: Moving averages, RSI, and other momentum indicators
- **Forecasting Models**: Auto-ARIMA implementation with configurable forecast periods
- **Risk Metrics**: Daily price changes, volatility calculations, and statistical summaries

## External Dependencies

### Financial Data APIs
- **Yahoo Finance (yfinance)**: Primary data source for stock prices, company information, and market data
- **Market Data**: Real-time and historical stock prices, trading volumes, and company fundamentals

### Python Libraries
- **Web Framework**: Flask for HTTP routing and template rendering
- **Data Science Stack**: 
  - pandas: Data manipulation and analysis
  - numpy: Numerical computing
  - plotly: Interactive data visualization
- **Time Series Analysis**:
  - pmdarima: Automated ARIMA model selection and fitting
  - statsmodels: Statistical modeling and time series analysis
  - scikit-learn: Machine learning utilities
- **Caching**: joblib for persistent function result caching
- **Utilities**: python-dateutil for date parsing and manipulation

### Deployment Dependencies
- **WSGI Server**: Gunicorn for production deployment
- **Development**: Flask's built-in development server for local testing

### Optional Integrations
- **News APIs**: Framework ready for financial news integration
- **Portfolio Management**: Architecture supports multi-stock comparison and portfolio tracking
- **Additional Data Sources**: Extensible design allows for integration with other financial data providers