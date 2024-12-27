from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from mlib.core.engine import Engine
from mlib.core.exchange import Exchange
from mlib.core.base_agent import BaseAgent
from exchange_config import ExchangeConfig
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 获取数据
def get_stock_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

# 数据预处理
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

# 特征工程
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df.dropna(inplace=True)
    return df

# 计算RSI
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 模型训练
def train_model(df: pd.DataFrame):
    if df.empty:
        raise ValueError("DataFrame is empty after feature engineering.")
    X = df[['MA_5', 'MA_10', 'RSI']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

# 初始化交易引擎
exchange_config = ExchangeConfig(
    fee=0.001,
    market_open="09:30",
    market_close="16:00",
    symbols=["AAPL"]
)
exchange = Exchange(exchange_config)
engine = Engine(exchange=exchange, description="Stock Prediction Engine", verbose=True)

# 添加代理
agent = BaseAgent()
agent.agent_id = 1
engine.agents[agent.agent_id] = agent

# 获取并处理数据
symbol = "AAPL"
df = get_stock_data(f"/Users/jerryli/Workspace/MarS/data/{symbol}.csv")
df = preprocess_data(df)
df = feature_engineering(df)

# 训练模型
model = train_model(df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    latest_data = df.iloc[-1][['MA_5', 'MA_10', 'RSI']].values.reshape(1, -1)
    latest_data_df = pd.DataFrame(latest_data, columns=['MA_5', 'MA_10', 'RSI'])
    predicted_price = model.predict(latest_data_df)
    return jsonify({'predicted_price': predicted_price[0]})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        global df, model
        df = get_stock_data(filepath)
        df = preprocess_data(df)
        df = feature_engineering(df)
        model = train_model(df)
        return '', 204  # No Content response to indicate success

if __name__ == '__main__':
    app.run(debug=True)
