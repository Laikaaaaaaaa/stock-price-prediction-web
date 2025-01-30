import tensorflow as tf

# Cấu hình bộ nhớ GPU trước khi sử dụng TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print("Error:", e)

# Các import khác
from flask import Flask, render_template, request
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime, timedelta
import os

# Định nghĩa các giá trị cần thiết
sequence_length = 100
num_features = 1
batch_size = 32
time_steps = 100
features = 1

# Mô hình LSTM
model = tf.keras.Sequential([
    LSTM(units=64, input_shape=(sequence_length, num_features), return_sequences=True),
    LSTM(units=64, return_sequences=True),
    LSTM(units=64),
    Dense(units=1)
])

# Kiểm tra số lượng GPU có sẵn
physical_devices = tf.config.list_physical_devices('GPU')
print(f'Num GPUs Available: {len(physical_devices)}')

#kiểm tra phiên bản tensorflow
print(tf.__version__)

if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# In thông tin chi tiết về các GPU có sẵn
for device in physical_devices:
    print(device)

# Kiểm tra và cấu hình GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Hạn chế TensorFlow sử dụng toàn bộ bộ nhớ GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

physical_devices = tf.config.list_physical_devices('GPU')
print("Các thiết bị GPU:", physical_devices)

if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Không tìm thấy GPU.")

# Thiết lập môi trường để không sử dụng GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Mức độ log đầy đủ

app = Flask(__name__)

# Kiểm tra phiên bản TensorFlow và GPU
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# In thông tin chi tiết về các GPU có sẵn
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)

# Load model
try:
    model = load_model('stock_dl_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock = request.form.get("stock")

        # Hàm tìm ngày gần nhất có dữ liệu giao dịch
        def get_valid_date():
            today = datetime.now()
            if today.weekday() in [5, 6]:  # 5 = Thứ 7, 6 = Chủ Nhật
                # Nếu hôm nay là thứ 7 hoặc Chủ Nhật, lấy ngày giao dịch của thứ Sáu
                target_date = today - timedelta(days=today.weekday() - 4 if today.weekday() == 5 else 2)
            elif today.weekday() == 0:  # Thứ 2
                # Nếu hôm nay là thứ 2, lấy ngày giao dịch của thứ Sáu trước đó
                target_date = today - timedelta(days=3)
            else:
                # Lấy ngày hôm nay nếu là ngày trong tuần (Thứ 2 - Thứ 6)
                target_date = today

            return target_date.strftime('%Y-%m-%d')

        # Lấy ngày cuối cùng có dữ liệu
        valid_date = get_valid_date()

        # Lấy dữ liệu từ Yahoo Finance
        try:
            df = yf.download(stock, start="2010-01-01", end=valid_date)
        except Exception as e:
            return render_template('index.html', error_message=f"Error downloading data: {str(e)}")

        # Tiền xử lý dữ liệu
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_data = df['Close'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(close_data)

        # Tạo dữ liệu cho mô hình
        x_data, y_data = [], []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i-100:i, 0])
            y_data.append(scaled_data[i, 0])
        x_data, y_data = np.array(x_data), np.array(y_data)

        # Chuyển đổi dữ liệu thành dạng phù hợp với mô hình LSTM
        x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

        # Dự đoán giá
        if model:
            try:
                y_predicted = model.predict(x_data)
                y_predicted = scaler.inverse_transform(y_predicted)

                # Dự đoán cho ngày tiếp theo
                last_100_days = scaled_data[-100:]
                last_100_days = np.reshape(last_100_days, (1, 100, 1))
                predicted_next_day = model.predict(last_100_days)
                predicted_next_day = scaler.inverse_transform(predicted_next_day)
                predicted_next_day_value = predicted_next_day[0][0]
            except Exception as e:
                return render_template('index.html', error_message=f"Error predicting stock prices: {str(e)}")
        else:
            return render_template('index.html', error_message="Model is not loaded properly.")

        # Tính các đường EMA
        ema20 = df['Close'].ewm(span=20, adjust=False).mean()
        ema50 = df['Close'].ewm(span=50, adjust=False).mean()
        ema100 = df['Close'].ewm(span=100, adjust=False).mean()
        ema200 = df['Close'].ewm(span=200, adjust=False).mean()

        # Tính MACD (Moving Average Convergence Divergence)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        # Tính RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Biểu đồ 1: EMA 20 và EMA 50
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Close'], label='Giá đóng cửa', color='black')
        ax.plot(ema20, label='EMA 20', color='blue')
        ax.plot(ema50, label='EMA 50', color='red')
        ax.set_title("Giá Đóng Cửa So Với Thời Gian (EMA 20 và EMA 50 Ngày)")
        ax.set_xlabel("Thời Gian")
        ax.set_ylabel("Giá")
        ax.legend()
        ema_chart_path = "static/ema_20_50.png"
        fig.savefig(ema_chart_path)
        plt.close(fig)

        # Biểu đồ 2: Giá đóng cửa so với thời gian với EMA 100 và EMA 200
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, 'y', label='Giá đóng cửa')
        ax2.plot(ema100, 'b', label='EMA 100')
        ax2.plot(ema200, 'm', label='EMA 200')
        ax2.set_title("Giá Đóng Cửa So Với Thời Gian (EMA 100 và EMA 200 Ngày)")
        ax2.set_xlabel("Thời Gian")
        ax2.set_ylabel("Giá")
        ax2.legend()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)
        
        # Biểu đồ 3: Dự đoán so với xu hướng gốc
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(df.Close, label='Giá đóng cửa gốc')
        ax3.plot(df.index[100:], y_predicted, label='Dự đoán giá cổ phiếu', color='orange')
        ax3.set_title("Dự Đoán So Với Xu Hướng Gốc")
        ax3.set_xlabel("Thời Gian")
        ax3.set_ylabel("Giá")
        ax3.legend()
        prediction_chart_path = "static/prediction_chart.png"
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)

        # Biểu đồ 4: MACD và Signal
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        ax5.plot(macd, label='MACD', color='blue')
        ax5.plot(signal, label='Signal', color='red')
        ax5.set_title('MACD và Signal')
        ax5.legend()
        macd_chart_path = "static/macd_chart.png"
        fig5.savefig(macd_chart_path)
        plt.close(fig5)

        # Biểu đồ 5: RSI
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        ax6.plot(rsi, label='RSI', color='purple')
        ax6.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        ax6.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        ax6.set_title('RSI - Relative Strength Index')
        ax6.legend()
        rsi_chart_path = "static/rsi_chart.png"
        fig6.savefig(rsi_chart_path)
        plt.close(fig6)

        # Lấy dữ liệu tuần từ thứ Hai đầu tuần đến thời gian hiện tại
        today = datetime.now()
        monday = today - timedelta(days=today.weekday())
        df_weekly = yf.download(stock, start=monday.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'), interval="1d")

        # Kiểm tra dữ liệu rỗng
        if df_weekly.empty:
            return render_template('index.html', error_message="Không có dữ liệu để hiển thị biểu đồ tuần.")

        # Tính các đường EMA cho dữ liệu tuần
        ema20_weekly = df_weekly['Close'].ewm(span=20, adjust=False).mean()
        ema50_weekly = df_weekly['Close'].ewm(span=50, adjust=False).mean()

        # Biểu đồ 6: EMA 20 và EMA 50 cho dữ liệu tuần
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.plot(df_weekly['Close'], label='Giá đóng cửa', color='black')
        ax4.plot(ema20_weekly, label='EMA 20 tuần', color='blue')
        ax4.plot(ema50_weekly, label='EMA 50 tuần', color='red')
        ax4.set_title("EMA 20 và EMA 50 cho Dữ Liệu Tuần")
        ax4.set_xlabel("Thời Gian")
        ax4.set_ylabel("Giá")
        ax4.legend()
        ema_chart_path_weekly = "static/ema_weekly.png"
        fig4.savefig(ema_chart_path_weekly)
        plt.close(fig4)

        return render_template(
            'index.html',
            stock_symbol=stock,
            predicted_price=predicted_next_day_value,
            plot_path_ema_20_50=ema_chart_path,
            plot_path_ema_100_200=ema_chart_path_100_200,
            plot_path_prediction=prediction_chart_path,
            plot_path_ema_weekly=ema_chart_path_weekly,
            plot_path_macd=macd_chart_path,
            plot_path_rsi=rsi_chart_path,
            data_desc=df.describe().to_html(classes='table table-bordered'),
            dataset_link="https://finance.yahoo.com/quote/" + stock
        )

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
