import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from mlib.core.engine import Engine
from mlib.core.exchange import Exchange
from mlib.core.base_agent import BaseAgent
from mlib.core.event import MarketOpenEvent
from exchange_config import ExchangeConfig  # 导入配置类

# 获取数据
def get_stock_data(symbol: str) -> pd.DataFrame:
    return pd.read_csv(f"/Users/jerryli/Workspace/MarS/data/{symbol}.csv")

# 数据预处理
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

# 特征工程
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['MA_5'] = df['Close'].rolling(window=5).mean()  # 调整窗口大小
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
df = get_stock_data(symbol)
print("Original Data:")
print(df)

df = preprocess_data(df)
print("After Preprocessing:")
print(df)

df = feature_engineering(df)
print("After Feature Engineering:")
print(df)

# 检查数据集大小
if df.empty:
    raise ValueError("DataFrame is empty after feature engineering.")

# 训练模型
model = train_model(df)

# 模拟交易
event = MarketOpenEvent(time=pd.Timestamp.now())  # 提供 time 参数
engine.events.append(event)

while engine.has_event():
    event = engine.events.pop(0)
    # 使用模型进行预测并执行交易逻辑
    # 例如：预测下一时刻的价格，根据预测结果决定买卖操作
    if isinstance(event, MarketOpenEvent):
        # 获取最新的特征数据
        latest_data = df.iloc[-1][['MA_5', 'MA_10', 'RSI']].values.reshape(1, -1)
        latest_data_df = pd.DataFrame(latest_data, columns=['MA_5', 'MA_10', 'RSI'])
        predicted_price = model.predict(latest_data_df)
        print(f"Predicted next close price: {predicted_price[0]}")
