# AI Trading Bot

This project is an AI-powered trading bot designed to perform simulated trading based on real-time market data. It utilizes a multi-agent system to analyze market signals from both quantitative and qualitative sources to make autonomous trading decisions.

## ✨ Features

- **🤖 Modular Agent-Based System**:
  - **`NewsAgent`**: Fetches and analyzes news sentiment to provide qualitative signals.
  - **`QuantitativeAgent`**: Performs technical analysis on stock data to generate quantitative signals.
  - **`AutonomousAgent`**: The core decision-making agent that can execute trades based on signals from other agents. It manages its own budget and portfolio.
- **📈 Real-time Interactive Dashboard**:
  - Built with Plotly Dash and styled with Tailwind CSS.
  - Visualizes portfolio value and Profit and Loss (PnL) in real-time.
  - Allows toggling the `AutonomousAgent` on and off.
  - Displays a live-updating chart of trading performance.
  - Shows a detailed history of all trades executed.
- **⚙️ Configurable**: Easily configure API keys, trading parameters like the stock ticker, initial capital, and agent settings through a central `config.py` and `.env` file.
- **🐍 Python-based**: Built with modern Python libraries including `yfinance` for market data, `pandas` for data manipulation, and `dash` for the web interface.

## 🛠️ Setup and Installation

Follow these steps to get your local development environment set up.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd trading-ai-project
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages. You should create a `requirements.txt` file for this.

```bash
pip install pandas yfinance dash plotly python-dotenv
# Or if a requirements.txt file is available:
# pip install -r requirements.txt
```

### 4. Set Up Environment Variables

The application uses a `.env` file to manage API keys and other secrets.

1.  Create a file named `.env` in the root of the project.
2.  Add your API keys to this file. You can get them from newsapi.org for news data. The trading API keys are for future integration with a real brokerage.

```env
# .env file
NEWS_API_KEY="your_news_api_key_here"

# For future brokerage integration
TRADING_API_KEY="your_trading_api_key_here"
TRADING_API_SECRET="your_trading_api_secret_here"
```

## ⚙️ Configuration

You can customize the bot's behavior by editing the `config.py` file:

- **`TICKER`**: The stock ticker to trade (e.g., `"RELIANCE.NS"`).
- **`CAPITAL`**: The initial capital for the main portfolio.
- **`CHECK_INTERVAL_SECONDS`**: How often the bot checks for new data and signals.
- **`QUANT_SHORT_WINDOW` / `QUANT_LONG_WINDOW`**: Moving average windows for the quantitative agent.
- **`AUTONOMOUS_ENABLED`**: Set to `True` to have the autonomous agent start trading automatically.
- **`AUTONOMOUS_BUDGET`**: The budget allocated to the autonomous agent.
- **`THRESHOLD_PROFIT_PERCENT`**: The profit percentage at which the agent will sell its position.

## 🚀 How to Run

1.  **Start the Trading Bot**:
    Run the main script from the root directory:
    ```bash
    python main.py
    ```
    This will initialize the agents and start the live trading simulation loop.

2.  **Access the Dashboard**:
    Open your web browser and go to:
    ```
    http://127.0.0.1:8050
    ```
    The dashboard will start and connect to the running bot, displaying real-time information. From the dashboard, you can enable or disable the autonomous agent.

## Project Structure

```
trading-ai-project/
├── .env                # Stores API keys and secrets (must be created)
├── config.py           # Main configuration for the bot and agents
├── main.py             # Main entry point to run the trading bot
├── dashboard.py        # Dash application for the web dashboard
├── news_agent.py       # Agent for news sentiment analysis
├── quantitative_agent.py # Agent for quantitative analysis
├── autonomous_agent.py # Agent for autonomous trading decisions
└── README.md
```

## 🔮 Future Improvements

- Integration with a real brokerage API for live trading.
- More sophisticated trading strategies and risk management models.
- A real-time chat interface for manual trade overrides (as hinted in `main.py`).
- Enhanced data sources beyond `yfinance` and basic news APIs.
