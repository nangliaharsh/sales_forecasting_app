import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import random

# Set page config
st.set_page_config(
    page_title="Sales Forecasting with Generative AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {}

# Helper Functions
def generate_synthetic_sales_data(start_date, end_date, product_category="Electronics"):
    """Generate synthetic sales data with realistic patterns"""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base trend with seasonality
    base_sales = 1000 + np.sin(np.arange(len(date_range)) * 2 * np.pi / 365) * 200
    
    # Weekly seasonality (higher sales on weekends)
    weekly_pattern = np.where(np.array([d.weekday() for d in date_range]) >= 5, 1.3, 1.0)
    
    # Monthly seasonality (holiday effects)
    monthly_pattern = np.where(np.array([d.month for d in date_range]) == 12, 1.5, 1.0)
    monthly_pattern *= np.where(np.array([d.month for d in date_range]) == 11, 1.2, 1.0)
    
    # Random noise
    noise = np.random.normal(0, 50, len(date_range))
    
    # Combine all factors
    sales = base_sales * weekly_pattern * monthly_pattern + noise
    sales = np.maximum(sales, 100)  # Ensure positive sales
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'sales': sales.astype(int),
        'product_category': product_category,
        'day_of_week': [d.strftime('%A') for d in date_range],
        'month': [d.strftime('%B') for d in date_range],
        'is_weekend': [d.weekday() >= 5 for d in date_range],
        'is_holiday_month': [d.month in [11, 12] for d in date_range]
    })
    
    return df

def generate_forecast(historical_data, forecast_days=30):
    """Generate sales forecast using trend analysis and seasonality"""
    # Simple trend analysis
    recent_data = historical_data.tail(30)
    trend = np.polyfit(range(len(recent_data)), recent_data['sales'], 1)[0]
    
    # Seasonal patterns
    avg_by_weekday = historical_data.groupby('day_of_week')['sales'].mean()
    seasonal_multiplier = historical_data.groupby(historical_data['date'].dt.month)['sales'].mean()
    
    # Generate forecast dates
    last_date = historical_data['date'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
    
    # Generate forecast
    base_forecast = recent_data['sales'].mean()
    forecast_values = []
    
    for i, date in enumerate(forecast_dates):
        # Apply trend
        trend_value = base_forecast + (trend * i)
        
        # Apply weekly seasonality
        weekday_mult = avg_by_weekday[date.strftime('%A')] / historical_data['sales'].mean()
        
        # Apply monthly seasonality
        month_mult = seasonal_multiplier.get(date.month, 1.0) / historical_data['sales'].mean()
        
        # Add some randomness
        random_factor = np.random.normal(1, 0.05)
        
        forecast_value = trend_value * weekday_mult * month_mult * random_factor
        forecast_values.append(max(forecast_value, 100))
    
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'predicted_sales': forecast_values,
        'confidence_lower': [v * 0.8 for v in forecast_values],
        'confidence_upper': [v * 1.2 for v in forecast_values]
    })
    
    return forecast_df

def generate_ai_insights(historical_data, forecast_data):
    """Generate AI-powered insights about the forecast"""
    insights = []
    
    # Trend analysis
    recent_avg = historical_data.tail(30)['sales'].mean()
    forecast_avg = forecast_data['predicted_sales'].mean()
    trend_change = ((forecast_avg - recent_avg) / recent_avg) * 100
    
    if trend_change > 5:
        insights.append(f"📈 Positive Trend: Sales are expected to increase by {trend_change:.1f}% based on current patterns.")
    elif trend_change < -5:
        insights.append(f"📉 Declining Trend: Sales may decrease by {abs(trend_change):.1f}%. Consider promotional activities.")
    else:
        insights.append("📊 Stable Trend: Sales are expected to remain relatively stable.")
    
    # Seasonality insights
    weekend_sales = historical_data[historical_data['is_weekend']]['sales'].mean()
    weekday_sales = historical_data[~historical_data['is_weekend']]['sales'].mean()
    weekend_uplift = ((weekend_sales - weekday_sales) / weekday_sales) * 100
    
    if weekend_uplift > 10:
        insights.append(f"🎯 Weekend Opportunity: Weekend sales are {weekend_uplift:.1f}% higher. Optimize weekend promotions.")
    
    # Holiday insights
    holiday_sales = historical_data[historical_data['is_holiday_month']]['sales'].mean()
    regular_sales = historical_data[~historical_data['is_holiday_month']]['sales'].mean()
    holiday_uplift = ((holiday_sales - regular_sales) / regular_sales) * 100
    
    if holiday_uplift > 20:
        insights.append(f"🎄 Holiday Boost: Holiday months show {holiday_uplift:.1f}% higher sales. Plan inventory accordingly.")
    
    # Volatility insight
    volatility = historical_data['sales'].std() / historical_data['sales'].mean()
    if volatility > 0.3:
        insights.append("⚠️ High Volatility: Sales show significant fluctuations. Consider demand smoothing strategies.")
    
    return insights

# Main App
def main():
    st.markdown('<h1 class="main-header">Sales Forecasting with Generative AI</h1>', unsafe_allow_html=True)
    
    # Sidebar for controls
    st.sidebar.header("📊 Control Panel")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Generate Synthetic Data", "Upload CSV File"]
    )
    
    if data_source == "Generate Synthetic Data":
        # Parameters for synthetic data
        product_category = st.sidebar.selectbox(
            "Product Category",
            ["Electronics", "Clothing", "Food & Beverages", "Home & Garden", "Sports"]
        )
        
        # Date range
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.date(2023, 1, 1))
        with col2:
            end_date = st.date_input("End Date", datetime.date(2024, 1, 31))
        
        if st.sidebar.button("Generate Data", type="primary"):
            with st.spinner("Generating synthetic sales data..."):
                st.session_state.historical_data = generate_synthetic_sales_data(start_date, end_date, product_category)
    
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Sales Data CSV", type=['csv'])
        if uploaded_file is not None:
            st.session_state.historical_data = pd.read_csv(uploaded_file)
            st.session_state.historical_data['date'] = pd.to_datetime(st.session_state.historical_data['date'])
    
    # Forecasting parameters
    st.sidebar.subheader("🔮 Forecast Settings")
    forecast_days = st.sidebar.slider("Forecast Period (days)", 7, 90, 30)
    confidence_level = st.sidebar.slider("Confidence Level (%)", 80, 99, 95)
    
    # Main content
    if 'historical_data' in st.session_state:
        historical_data = st.session_state.historical_data
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Records",
                len(historical_data),
                delta=f"{len(historical_data)} days"
            )
        
        with col2:
            avg_daily_sales = historical_data['sales'].mean()
            st.metric(
                "Avg Daily Sales",
                f"${avg_daily_sales:,.0f}",
                delta=f"{(avg_daily_sales/1000):.1f}K"
            )
        
        with col3:
            total_sales = historical_data['sales'].sum()
            st.metric(
                "Total Sales",
                f"${total_sales:,.0f}",
                delta=f"{(total_sales/1000000):.1f}M"
            )
        
        with col4:
            growth_rate = ((historical_data['sales'].tail(30).mean() / historical_data['sales'].head(30).mean()) - 1) * 100
            st.metric(
                "Growth Rate",
                f"{growth_rate:+.1f}%",
                delta="vs. first 30 days"
            )
        
        # Generate forecast
        if st.button("Generate Forecast", type="primary", use_container_width=True):
            with st.spinner("Generating AI-powered forecast..."):
                forecast_data = generate_forecast(historical_data, forecast_days)
                st.session_state.forecast_data = forecast_data
        
        # Display results
        if st.session_state.forecast_data is not None:
            forecast_data = st.session_state.forecast_data
            
            # Forecast visualization
            st.subheader("📈 Sales Forecast Visualization")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Sales Forecast', 'Historical vs Predicted', 'Daily Patterns', 'Confidence Intervals'),
                specs=[[{"colspan": 2}, None], [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Main forecast plot
            fig.add_trace(
                go.Scatter(
                    x=historical_data['date'].tail(60),
                    y=historical_data['sales'].tail(60),
                    mode='lines',
                    name='Historical Sales',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_data['date'],
                    y=forecast_data['predicted_sales'],
                    mode='lines+markers',
                    name='Predicted Sales',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            # Confidence intervals
            fig.add_trace(
                go.Scatter(
                    x=forecast_data['date'],
                    y=forecast_data['confidence_upper'],
                    mode='lines',
                    name='Upper Confidence',
                    line=dict(color='lightgray', width=1),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_data['date'],
                    y=forecast_data['confidence_lower'],
                    mode='lines',
                    name='Lower Confidence',
                    line=dict(color='lightgray', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.2)',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # Daily patterns
            daily_avg = historical_data.groupby('day_of_week')['sales'].mean()
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_avg = daily_avg.reindex(weekday_order)
            
            fig.add_trace(
                go.Bar(
                    x=daily_avg.index,
                    y=daily_avg.values,
                    name='Daily Average',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=800, showlegend=True, title_text="Comprehensive Sales Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Insights
            st.subheader("🤖 AI-Generated Insights")
            insights = generate_ai_insights(historical_data, forecast_data)
            
            for insight in insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
            
            # Scenario Planning
            st.subheader("🎯 Scenario Planning")
            
            scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
            
            with scenario_col1:
                st.write("**Optimistic Scenario (+20%)**")
                optimistic = forecast_data['predicted_sales'] * 1.2
                opt_total = optimistic.sum()
                st.metric("Forecasted Revenue", f"${opt_total:,.0f}", delta="+20%")
            
            with scenario_col2:
                st.write("**Base Scenario**")
                base_total = forecast_data['predicted_sales'].sum()
                st.metric("Forecasted Revenue", f"${base_total:,.0f}", delta="Base")
            
            with scenario_col3:
                st.write("**Conservative Scenario (-15%)**")
                conservative = forecast_data['predicted_sales'] * 0.85
                cons_total = conservative.sum()
                st.metric("Forecasted Revenue", f"${cons_total:,.0f}", delta="-15%")
            
            # Recommendations
            st.subheader("💡 Strategic Recommendations")
            
            recommendations = [
                "📦 **Inventory Planning**: Stock levels should be adjusted based on the {:.0f}% forecasted change".format(growth_rate),
                "🎯 **Marketing Focus**: Increase promotional activities during predicted low-sales periods",
                "📊 **Performance Monitoring**: Track actual vs. predicted sales to improve model accuracy",
                "🔄 **Dynamic Pricing**: Consider price adjustments during high-demand forecast periods",
                "🚀 **Capacity Planning**: Prepare logistics and staffing for forecasted sales volumes"
            ]
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Export options
            st.subheader("📥 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # Prepare export data
                export_data = forecast_data.copy()
                export_data['historical_sales'] = None
                
                # Add last few days of historical data
                recent_historical = historical_data.tail(30)[['date', 'sales']].copy()
                recent_historical.columns = ['date', 'historical_sales']
                recent_historical['predicted_sales'] = None
                recent_historical['confidence_lower'] = None
                recent_historical['confidence_upper'] = None
                
                combined_export = pd.concat([recent_historical, export_data], ignore_index=True)
                csv_data = combined_export.to_csv(index=False)
                
                st.download_button(
                    label="Download Forecast CSV",
                    data=csv_data,
                    file_name=f"sales_forecast_{datetime.date.today()}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Generate summary report
                report_data = {
                    "forecast_period": f"{forecast_days} days",
                    "total_predicted_sales": f"${base_total:,.0f}",
                    "average_daily_sales": f"${forecast_data['predicted_sales'].mean():,.0f}",
                    "confidence_level": f"{confidence_level}%",
                    "growth_trend": f"{growth_rate:+.1f}%"
                }
                
                st.download_button(
                    label="Download Summary Report",
                    data=str(report_data),
                    file_name=f"forecast_summary_{datetime.date.today()}.txt",
                    mime="text/plain"
                )
    
    else:
        st.info("👆 Please generate synthetic data or upload a CSV file to begin forecasting.")
        
        # Show sample data format
        st.subheader("📋 Required Data Format")
        sample_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'sales': [1200, 1350, 1180],
            'product_category': ['Electronics', 'Electronics', 'Electronics']
        })
        st.dataframe(sample_data)
        
        st.markdown("""
        **Required Columns:**
        - `date`: Date in YYYY-MM-DD format
        - `sales`: Numeric sales values
        - `product_category`: Product category (optional)
        """)

if __name__ == "__main__":
    main()