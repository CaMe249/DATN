# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Walmart Stock Forecast", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Walmart_Stock.csv", parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    return df, min_date, max_date

def forecast_holt_winters(df, steps):
    model = ExponentialSmoothing(df['Close'], trend='add', seasonal=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps)
    return forecast, model_fit

def show_header():
    st.markdown("""
        <style>
            .header-container {
                display: flex;
                align-items: center;
                justify-content: space-between;
                background-color: white;
                padding: 10px 20px;
                border-bottom: 1px solid #ddd;
            }
            .header-left {
                display: flex;
                align-items: center;
            }
            .header-left img {
                width: 100px;
            }
            .header-left h1 {
                font-size: 40px;
                margin-left: 15px;
            }
            .nav-menu a {
                margin: 0 15px;
                text-decoration: none;
                font-size: 18px;
                color: black;
                transition: all 0.2s ease-in-out;
            }
            .nav-menu a:hover {
                font-size: 20px;
                font-weight: bold;
            }
        </style>
        <div class='header-container'>
            <div class='header-left'>
                <img src='https://1000logos.net/wp-content/uploads/2017/05/Walmart-Logo-768x432.png'>
                <h1>Walmart Inc.</h1>
            </div>
            <div class='nav-menu'>
                <a href='/?page=home' target='_self'>Trang chủ</a>
                <a href='/?page=thong-ke' target='_self'>Thống kê mô tả</a>
                <a href='/?page=du-doan' target='_self'>Dự báo cổ phiếu</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

def get_selected_page():
    query_params = st.query_params
    return query_params.get("page", "home")

def page_home(df, min_date, max_date):
    st.subheader("Chọn khoảng thời gian hiển thị:")
    start_date = st.date_input("Từ ngày", value=min_date, min_value=min_date, max_value=max_date, key="home_start")
    end_date = st.date_input("Đến ngày", value=max_date, min_value=min_date, max_value=max_date, key="home_end")

    if start_date > end_date:
        st.warning("Ngày bắt đầu phải nhỏ hơn ngày kết thúc!")
        return

    filtered_df = df.loc[start_date:end_date]
    st.subheader("Dữ liệu gốc trong khoảng thời gian đã chọn:")
    st.dataframe(filtered_df)

def page_thong_ke(df, min_date, max_date):
    st.title("Thống kê mô tả")

    st.subheader("Chọn khoảng thời gian phân tích:")
    start_date = st.date_input("Từ ngày", value=min_date, min_value=min_date, max_value=max_date, key="stat_start")
    end_date = st.date_input("Đến ngày", value=max_date, min_value=min_date, max_value=max_date, key="stat_end")

    if start_date > end_date:
        st.warning("Ngày bắt đầu phải nhỏ hơn ngày kết thúc!")
        return

    filtered_df = df.loc[start_date:end_date]

    st.subheader("Biểu đồ nến ")
    fig = go.Figure(data=[
        go.Candlestick(x=filtered_df.index,
                       open=filtered_df['Open'],
                       high=filtered_df['High'],
                       low=filtered_df['Low'],
                       close=filtered_df['Close'])
    ])
    fig.update_layout(xaxis_title='Ngày', yaxis_title='Giá', height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Biểu đồ giá đóng cửa trung bình theo năm")
    yearly_close = filtered_df['Close'].resample('Y').mean().reset_index()
    fig_bar = px.bar(yearly_close, x='Date', y='Close', labels={'Close': 'Giá trung bình', 'Date': 'Năm'})
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Biểu đồ khối lượng giao dịch theo thời gian")
    fig_volume = px.line(filtered_df.reset_index(), x='Date', y='Volume', title='Khối lượng giao dịch')
    st.plotly_chart(fig_volume, use_container_width=True)

def page_du_doan(df):
    st.title("Dự báo giá cổ phiếu Walmart")
    option = st.selectbox("Chọn khoảng thời gian dự đoán:", ("Tuần (7 ngày)", "Tháng (30 ngày)", "Năm (365 ngày)"))
    steps = 7 if "Tuần" in option else 30 if "Tháng" in option else 365

    forecast, model_fit = forecast_holt_winters(df, steps)
    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    forecast_df = pd.DataFrame({"Ngày": forecast_index, "Giá dự báo": forecast.values})

    st.subheader("Bảng kết quả dự đoán:")
    st.dataframe(forecast_df.set_index("Ngày"))

    st.subheader("Đánh giá sai số (trên tập huấn luyện)")

    fitted_values = model_fit.fittedvalues
    actual = df['Close'][-len(fitted_values):]
    mae = mean_absolute_error(actual, fitted_values)
    mse = mean_squared_error(actual, fitted_values)
    rmse = np.sqrt(mse)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Sai số tuyệt đối trung bình)", f"{mae:.2f}")
    col2.metric("MSE (Sai số bình phương trung bình)", f"{mse:.2f}")
    col3.metric("RMSE (Căn bậc hai của MSE)", f"{rmse:.2f}")

    st.subheader("Biểu đồ dự đoán giá đóng cửa")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['Close'], label='Giá thực tế')
    ax.plot(forecast_index, forecast, label='Giá dự báo', color='red')
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá đóng cửa")
    ax.legend()
    st.pyplot(fig)

# MAIN
show_header()
df, min_date, max_date = load_data()
selected = get_selected_page()

if selected == "home":
    page_home(df, min_date, max_date)
elif selected == "thong-ke":
    page_thong_ke(df, min_date, max_date)
elif selected == "du-doan":
    page_du_doan(df)
