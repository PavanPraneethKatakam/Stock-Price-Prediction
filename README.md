
# 📈 Stock Price Prediction and Analysis

This project is a comprehensive and interactive tool for stock price prediction using machine learning and deep learning models. Built with Python and deployed using Streamlit, the app allows users to analyze historical stock data, visualize technical indicators, and forecast future prices with confidence intervals.

---

## 🔧 Technologies Used

- **Python**  
- **Pandas, NumPy** – Data manipulation and analysis  
- **Matplotlib, Seaborn, Plotly** – Data visualization  
- **yfinance** – Historical stock data retrieval  
- **scikit-learn** – Machine Learning models  
- **XGBoost** – Gradient boosting  
- **Prophet (Facebook)** – Time series forecasting  
- **TensorFlow (Keras)** – LSTM deep learning model  
- **Statsmodels** – ARIMA model  
- **Streamlit** – Web application framework  
- **TextBlob** – Sentiment analysis  
- **BeautifulSoup** – Web scraping for sentiment/news  
- **SciPy** – Statistical processing  

---

## 📊 Features

- Fetch historical stock data using a ticker symbol
- Compute and visualize technical indicators: MA, RSI, MACD, etc.
- Engineer features such as price change, volume change, high-low diff, and sentiment
- Train models like:
  - Random Forest
  - LSTM
  - ARIMA
  - Prophet
- Predict stock prices for the next N days (user-defined)
- Display interactive plots for:
  - Historical prices
  - Predicted future prices
  - 95% confidence intervals
- View prediction statistics in a tabular format
- Includes a disclaimer about prediction risks

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stock-price-predictor.git
   cd stock-price-predictor
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

---

## 📁 File Structure

- `app.py` – Main Streamlit application
- `requirements.txt` – List of dependencies
- `README.md` – Project overview and instructions

---

## 📌 Note

- Predictions are based on historical data and may not represent future performance accurately.
- Sentiment scores are estimated and can vary based on news content.

---

## 📜 License

This project is open-source and available under the MIT License.
