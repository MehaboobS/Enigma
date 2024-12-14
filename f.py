import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import google.generativeai as genai

# API Key for Twelve Data
YOUR_API = "6c68304f987b446994927af28357c4dd"
BASE_URL = f"https://api.twelvedata.com/time_series?apikey={YOUR_API}"

# Configure GenAI API
genai.configure(api_key="AIzaSyDswcY725_kLa1h01qrJgiqDPcTKe5Npmk")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize NLTK Sentiment Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Sidebar Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", ["Stock Price Predictor", "Sentiment Analysis", "AI Stock Chatbot"])

if options == "Stock Price Predictor":
    st.title("Stock Price Analysis")

    # Input fields for stock symbol and interval
    SYMBOL = st.text_input("Enter the stock symbol (e.g., IBM):", "IBM")
    INTERVAL = st.selectbox("Select the interval:", ["1min", "5min", "15min", "30min", "60min"])

    if st.button("Fetch and Analyze Stock Data"):
        # Construct the parameters for the API request
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "outputsize": "150",  # Fetch the last 30 data points
            "format": "JSON"
        }

        # Send the request to the Twelve Data API
        response = requests.get(BASE_URL, params=params)

        # Check for a successful response
        if response.status_code == 200:
            try:
                data = response.json()
                if "values" in data:
                    df = pd.DataFrame(data["values"])
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df['close'] = pd.to_numeric(df['close'], errors='coerce')
                    df['open'] = pd.to_numeric(df['open'], errors='coerce')
                    df['high'] = pd.to_numeric(df['high'], errors='coerce')
                    df['low'] = pd.to_numeric(df['low'], errors='coerce')

                    st.write(f"Data for {SYMBOL} at {INTERVAL} interval:")
                    st.write(df)

                    # Plotting the Candlestick chart
                    fig = go.Figure(data=[go.Candlestick(
                        x=df['datetime'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        increasing_line_color='green', decreasing_line_color='red', 
                    )])

                    fig.update_layout(
                        title=f"Stock Price for {SYMBOL} ({INTERVAL} interval)",
                        xaxis_title="Date/Time",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=False
                    )

                    st.plotly_chart(fig)

                else:
                    st.error("No data available for the selected symbol and interval.")

            except Exception as e:
                st.error(f"Error parsing data: {e}")
        else:
            st.error(f"Failed to fetch data. HTTP Status code: {response.status_code}. Response: {response.text}")

elif options == "Sentiment Analysis":
    st.title("Sentiment Analysis")
    st.write("Enter a text to analyze its sentiment.")

    user_input = st.text_input("Enter your text:", "")
    
    if user_input:
        # Function to fetch news for sentiment analysis
        def get_stock_news(stock_symbol):
            url = f"https://newsdata.io/api/1/latest?apikey=pub_62285e1ac368477b382a0bdccbc24e1a164f3&q={stock_symbol}&country=us&domainurl=news.google.com"
            response = requests.get(url)
            return response.json().get('results', [])

        def analyze_sentiment(news_articles):
            sentiments = []
            for article in news_articles:
                text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
                sentiment_score = sia.polarity_scores(text)
                sentiments.append((article.get('title', ''), sentiment_score))
            return sentiments

        # Fetch and analyze news sentiment
        articles = get_stock_news(user_input)
        sentiments = analyze_sentiment(articles)

        def display_sentiment_results(sentiments):
            positive, negative, neutral = 0, 0, 0
            for _, sentiment in sentiments:
                score = sentiment['compound']
                if score >= 0.05:
                    positive += 1
                elif score <= -0.05:
                    negative += 1
                else:
                    neutral += 1
            
            st.write(f"**Positive Articles:** {positive}")
            st.write(f"**Negative Articles:** {negative}")
            st.write(f"**Neutral Articles:** {neutral}")

        # Display sentiment results
        display_sentiment_results(sentiments)

elif options == "AI Stock Chatbot":
    st.title("AI-Powered Stock Analysis Chatbot")
    st.write("Ask the AI chatbot anything about the stock market.")

    # Chatbot toggle
    if "show_chatbot" not in st.session_state:
        st.session_state["show_chatbot"] = False

    if st.button("💬 Bot"):
        st.session_state["show_chatbot"] = not st.session_state["show_chatbot"]

    # Show chatbot interface if toggled
    if st.session_state["show_chatbot"]:
        user_input = st.text_input("Your query:", placeholder="Type your stock-related query here...")

        def is_stock_related(query):
            stock_keywords = ["stock", "market", "price", "shares", "investment", "trading", "IPO", "NASDAQ", "portfolio"]
            return any(keyword.lower() in query.lower() for keyword in stock_keywords)

        if user_input:
            if is_stock_related(user_input):
                with st.spinner("Generating response..."):
                    try:
                        response = model.generate_content(user_input)
                        st.success("Response generated!")
                        st.write(f"**Bot Response:** {response.text}")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("This chatbot only responds to stock market-related queries.")
