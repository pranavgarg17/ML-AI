import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

st.title('💰 Bitcoin Price Predictor with Visualizations')
st.write('Enter values to predict the Closing Price of Bitcoin and explore interactive visualizations')

@st.cache_resource
def load_models():
    try:
        lr_model=joblib.load('linear_regression_model.pkl')
        svr_model=joblib.load('svr_model.pkl')
        tree_model=joblib.load('tree_regression_model.pkl')
        scaler=joblib.load('scaler.pkl')
        return lr_model, svr_model, tree_model, scaler
    except:
        st.error('⚠️ Model files not in memory! Execute the model training code!!')
        return None, None, None, None

@st.cache_data
def generate_sample_data():
    """Generate sample Bitcoin data for visualization"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    # Generate realistic Bitcoin price data
    base_price = 30000
    price_trend = np.linspace(0, 10000, len(dates))
    noise = np.random.normal(0, 2000, len(dates))
    seasonal = 1000 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    
    close_prices = base_price + price_trend + noise + seasonal
    close_prices = np.maximum(close_prices, 15000)  # Ensure no negative prices
    
    # Generate other features based on close prices
    high_prices = close_prices * (1 + np.random.uniform(0.01, 0.05, len(dates)))
    low_prices = close_prices * (1 - np.random.uniform(0.01, 0.05, len(dates)))
    open_prices = close_prices + np.random.normal(0, 500, len(dates))
    volumes = np.random.lognormal(15, 0.5, len(dates))
    marketcaps = close_prices * np.random.uniform(19e6, 21e6, len(dates))
    
    return pd.DataFrame({
        'Date': dates,
        'High': high_prices,
        'Low': low_prices,
        'Open': open_prices,
        'Close': close_prices,
        'Volume': volumes,
        'Marketcap': marketcaps
    })

lr_model, svr_model, tree_model, scaler = load_models()

# Generate sample data for visualizations
sample_data = generate_sample_data()

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["🔮 Price Prediction", "📊 Data Visualization", "📈 Model Comparison"])

with tab1:
    if lr_model is not None:
        # Input section
        st.header("📉 Enter Bitcoin Data")

        col1, col2 = st.columns(2)

        with col1:
            high = st.number_input('High Price ($)', value=45000.00, min_value=0.0)
            low = st.number_input("Low Price ($)", value=42000.00, min_value=0.0)
            open_price = st.number_input("Open Price ($)", value=43500.00, min_value=0.0)
        
        with col2:
            volume = st.number_input("Volume", value=1000000.0, min_value=0.0)
            marketcap = st.number_input("Market Cap", value=850000000000.0, min_value=0.0)
        
        model_choice = st.selectbox("Choose Model:", ["Linear Regression", "SVR", 'Decision Tree'])

        if st.button('🔮 Predict Price'):
            input_data = np.array([[high, low, open_price, volume, marketcap]])
            input_scaled = scaler.transform(input_data)

            # Get predictions from all models
            lr_pred = lr_model.predict(input_scaled)[0]
            svr_pred = svr_model.predict(input_scaled)[0]
            tree_pred = tree_model.predict(input_scaled)[0]

            # Display selected model prediction
            if model_choice == 'Linear Regression':
                prediction = lr_pred
                st.success(f"🔵 Linear Regression Prediction: **${prediction:,.2f}**")
            elif model_choice == 'SVR':
                prediction = svr_pred
                st.success(f"🔵 SVR Prediction: **${prediction:,.2f}**")
            else:
                prediction = tree_pred
                st.success(f"🔵 Tree Prediction: **${prediction:,.2f}**")

            # Prediction comparison chart
            st.subheader("📊 Model Predictions Comparison")
            
            pred_df = pd.DataFrame({
                'Model': ['Linear Regression', 'SVR', 'Decision Tree'],
                'Prediction': [lr_pred, svr_pred, tree_pred]
            })
            
            fig = px.bar(pred_df, x='Model', y='Prediction', 
                        title='Price Predictions from Different Models',
                        color='Model',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            
            fig.update_layout(
                xaxis_title="Model",
                yaxis_title="Predicted Price ($)",
                showlegend=False
            )
            
            # Add value labels on bars
            for i, row in pred_df.iterrows():
                fig.add_annotation(
                    x=row['Model'],
                    y=row['Prediction'],
                    text=f"${row['Prediction']:,.0f}",
                    showarrow=False,
                    yshift=10
                )
            
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance visualization
            st.subheader("📈 Input Features Visualization")
            
            feature_df = pd.DataFrame({
                'Feature': ['High', 'Low', 'Open', 'Volume', 'MarketCap'],
                'Value': [high, low, open_price, volume, marketcap],
                'Normalized': [
                    (high - 20000) / 60000,
                    (low - 20000) / 60000,
                    (open_price - 20000) / 60000,
                    (volume - 500000) / 2000000,
                    (marketcap - 300000000000) / 1000000000000
                ]
            })
            
            fig_features = px.bar(feature_df, x='Feature', y='Normalized',
                                title='Normalized Input Features',
                                color='Normalized',
                                color_continuous_scale='viridis')
            
            st.plotly_chart(fig_features, use_container_width=True)

            # Input summary table
            st.subheader("📋 Your Inputs Summary")
            input_summary = pd.DataFrame({
                'Feature': ['High', 'Low', 'Open', 'Volume', 'MarketCap'],
                'Value': [f"${high:,.2f}", f"${low:,.2f}", f"${open_price:,.2f}", 
                         f"{volume:,.0f}", f"${marketcap:,.0f}"]
            })
            st.table(input_summary)

with tab2:
    st.header("📊 Bitcoin Data Visualization")
    
    # Price trend chart
    st.subheader("💹 Bitcoin Price Trend")
    
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=sample_data['Date'],
        y=sample_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=sample_data['Date'],
        y=sample_data['High'],
        mode='lines',
        name='High Price',
        line=dict(color='#4ECDC4', width=1, dash='dash')
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=sample_data['Date'],
        y=sample_data['Low'],
        mode='lines',
        name='Low Price',
        line=dict(color='#45B7D1', width=1, dash='dash')
    ))
    
    fig_trend.update_layout(
        title="Bitcoin Price Trend Over Time",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Candlestick chart
    st.subheader("🕯️ Bitcoin Candlestick Chart")
    
    # Sample last 90 days for candlestick
    recent_data = sample_data.tail(90)
    
    fig_candlestick = go.Figure(data=go.Candlestick(
        x=recent_data['Date'],
        open=recent_data['Open'],
        high=recent_data['High'],
        low=recent_data['Low'],
        close=recent_data['Close'],
        name='Bitcoin'
    ))
    
    fig_candlestick.update_layout(
        title="Bitcoin Candlestick Chart (Last 90 Days)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig_candlestick, use_container_width=True)
    
    # Volume analysis
    st.subheader("📊 Volume Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_volume = px.area(sample_data, x='Date', y='Volume',
                           title='Trading Volume Over Time',
                           color_discrete_sequence=['#9B59B6'])
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        # Volume vs Price correlation
        fig_scatter = px.scatter(sample_data, x='Volume', y='Close',
                               title='Volume vs Close Price',
                               color='Date',
                               color_continuous_scale='plasma')
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Price distribution
    st.subheader("📈 Price Distribution")
    
    fig_hist = px.histogram(sample_data, x='Close', nbins=50,
                          title='Bitcoin Close Price Distribution',
                          color_discrete_sequence=['#E67E22'])
    
    fig_hist.update_layout(
        xaxis_title="Close Price ($)",
        yaxis_title="Frequency"
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.header("📈 Model Performance Comparison")
    
    if lr_model is not None:
        # Generate sample predictions for comparison
        sample_inputs = sample_data[['High', 'Low', 'Open', 'Volume', 'Marketcap']].tail(100)
        sample_scaled = scaler.transform(sample_inputs)
        
        lr_predictions = lr_model.predict(sample_scaled)
        svr_predictions = svr_model.predict(sample_scaled)
        tree_predictions = tree_model.predict(sample_scaled)
        actual_prices = sample_data['Close'].tail(100).values
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Actual': actual_prices,
            'Linear Regression': lr_predictions,
            'SVR': svr_predictions,
            'Decision Tree': tree_predictions
        })
        
        # Predictions vs Actual
        st.subheader("🎯 Predictions vs Actual Prices")
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Scatter(
            y=comparison_df['Actual'],
            mode='lines',
            name='Actual',
            line=dict(color='black', width=3)
        ))
        
        fig_comparison.add_trace(go.Scatter(
            y=comparison_df['Linear Regression'],
            mode='lines',
            name='Linear Regression',
            line=dict(color='#FF6B6B', width=2)
        ))
        
        fig_comparison.add_trace(go.Scatter(
            y=comparison_df['SVR'],
            mode='lines',
            name='SVR',
            line=dict(color='#4ECDC4', width=2)
        ))
        
        fig_comparison.add_trace(go.Scatter(
            y=comparison_df['Decision Tree'],
            mode='lines',
            name='Decision Tree',
            line=dict(color='#45B7D1', width=2)
        ))
        
        fig_comparison.update_layout(
            title="Model Predictions vs Actual Prices",
            xaxis_title="Sample Index",
            yaxis_title="Price ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Error analysis
        st.subheader("📊 Error Analysis")
        
        # Calculate errors
        lr_errors = np.abs(actual_prices - lr_predictions)
        svr_errors = np.abs(actual_prices - svr_predictions)
        tree_errors = np.abs(actual_prices - tree_predictions)
        
        error_df = pd.DataFrame({
            'Model': ['Linear Regression', 'SVR', 'Decision Tree'],
            'Mean Absolute Error': [np.mean(lr_errors), np.mean(svr_errors), np.mean(tree_errors)],
            'RMSE': [np.sqrt(np.mean((actual_prices - lr_predictions)**2)),
                    np.sqrt(np.mean((actual_prices - svr_predictions)**2)),
                    np.sqrt(np.mean((actual_prices - tree_predictions)**2))]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_mae = px.bar(error_df, x='Model', y='Mean Absolute Error',
                           title='Mean Absolute Error by Model',
                           color='Model',
                           color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            st.plotly_chart(fig_mae, use_container_width=True)
        
        with col2:
            fig_rmse = px.bar(error_df, x='Model', y='RMSE',
                            title='Root Mean Square Error by Model',
                            color='Model',
                            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Error distribution
        st.subheader("📈 Error Distribution")
        
        error_dist_df = pd.DataFrame({
            'Linear Regression': lr_errors,
            'SVR': svr_errors,
            'Decision Tree': tree_errors
        })
        
        fig_error_dist = go.Figure()
        
        for model in error_dist_df.columns:
            fig_error_dist.add_trace(go.Box(
                y=error_dist_df[model],
                name=model,
                boxpoints='outliers'
            ))
        
        fig_error_dist.update_layout(
            title="Error Distribution by Model",
            yaxis_title="Absolute Error ($)"
        )
        
        st.plotly_chart(fig_error_dist, use_container_width=True)

# Enhanced sidebar
st.sidebar.header("ℹ️ Instructions")
st.sidebar.write("""
1. **Prediction Tab**: Enter Bitcoin data and get predictions
2. **Visualization Tab**: Explore historical data patterns
3. **Comparison Tab**: Compare model performance
4. Use interactive charts to zoom and explore data
""")

st.sidebar.header("📝 About Models")
st.sidebar.write("""
**Linear Regression**: Simple, interpretable, fast predictions

**SVR**: Support Vector Regression, good for non-linear patterns

**Decision Tree**: Captures complex relationships, prone to overfitting
""")

st.sidebar.header("📊 Visualization Features")
st.sidebar.write("""
- **Price Trends**: Historical price movements
- **Candlestick Charts**: OHLC visualization
- **Volume Analysis**: Trading volume patterns
- **Model Comparison**: Performance metrics
- **Error Analysis**: Model accuracy assessment
""")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Bitcoin Price Predictor with Advanced Visualizations")