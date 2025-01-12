from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime, timedelta

app = Flask(__name__)

# Load model
model = load_model('stock_dl_model.h5')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock = request.form.get("stock")

        # Lấy dữ liệu cổ phiếu từ Yahoo Finance
        df = yf.download(stock, start="2010-01-01", end=datetime.now().strftime('%Y-%m-%d'))
        
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
        y_predicted = model.predict(x_data)
        y_predicted = scaler.inverse_transform(y_predicted)

        # Dự đoán cho ngày tiếp theo
        last_100_days = scaled_data[-100:]
        last_100_days = np.reshape(last_100_days, (1, 100, 1))
        predicted_next_day = model.predict(last_100_days)
        predicted_next_day = scaler.inverse_transform(predicted_next_day)

        # Tính các đường EMA
        ema20 = df['Close'].ewm(span=20, adjust=False).mean()
        ema50 = df['Close'].ewm(span=50, adjust=False).mean()
        ema100 = df['Close'].ewm(span=100, adjust=False).mean()
        ema200 = df['Close'].ewm(span=200, adjust=False).mean()

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
        
        # Trả về kết quả dự đoán và các biểu đồ
        return render_template(
            'index.html',
            stock_symbol=stock,
            predicted_price=predicted_next_day,
            plot_path_ema_20_50=ema_chart_path,
            plot_path_ema_100_200=ema_chart_path_100_200,
            plot_path_prediction=prediction_chart_path,
            data_desc=df.describe().to_html(classes='table table-bordered'),
            dataset_link=f"/static/{stock}_data.csv"
        )
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
