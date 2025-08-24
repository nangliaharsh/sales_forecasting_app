# Sales Forecasting with Generative AI

This Streamlit app provides interactive sales forecasting using synthetic or uploaded sales data. It leverages generative AI techniques to analyze trends, seasonality, and volatility, and offers actionable insights and scenario planning for business decision-making.

## Features

- **Synthetic Data Generation:** Create realistic sales data with seasonality, holidays, and noise.
- **CSV Upload:** Import your own sales data for forecasting.
- **AI-Powered Forecast:** Predict future sales using trend and seasonal analysis.
- **Interactive Visualizations:** Explore historical and forecasted sales with Plotly charts.
- **Scenario Planning:** Compare optimistic, base, and conservative forecasts.
- **Strategic Recommendations:** Get AI-generated business insights and suggestions.
- **Export Options:** Download forecast results and summary reports.

## Usage

1. **Install dependencies:**
    ```sh
    pip install streamlit pandas numpy plotly
    ```

2. **Run the app:**
    ```sh
    streamlit run sales_forecasting_app.py
    ```

3. **Interact:**
    - Choose to generate synthetic data or upload a CSV file.
    - Adjust forecast period and confidence level.
    - Click "Generate Forecast" to view results and insights.
    - Download results as CSV or summary report.

## Data Format

If uploading your own CSV, ensure it contains:
- `date`: Date in YYYY-MM-DD format
- `sales`: Numeric sales values
- `product_category`: Product category (optional)

## Example

| date       | sales | product_category |
|------------|-------|-----------------|
| 2023-01-01 | 1200  | Electronics     |
| 2023-01-02 | 1350  | Electronics     |
| 2023-01-03 | 1180  | Electronics     |

## License

This project is provided for educational and demonstration purposes.

## Author

Harsh Nanglia