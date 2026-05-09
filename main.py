from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-5.4-mini"  # Use a different model
config["quick_think_llm"] = "gpt-5.4-mini"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds

# Configure data vendors (using AKShare)
config["data_vendors"] = {
    "core_stock_apis": "akshare",            # Options: alpha_vantage, yfinance, akshare
    "technical_indicators": "akshare",       # Options: alpha_vantage, yfinance, akshare
    "fundamental_data": "akshare",           # Options: alpha_vantage, yfinance, akshare
    "news_data": "akshare",                  # Options: alpha_vantage, yfinance, akshare
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("600519", "2026-05-07")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
