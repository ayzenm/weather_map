import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from joblib import Parallel, delayed

# Реальные средние температуры (примерные данные) для городов по сезонам
seasonal_temperatures = {
    "New York": {"winter": 0, "spring": 10, "summer": 25, "autumn": 15},
    "London": {"winter": 5, "spring": 11, "summer": 18, "autumn": 12},
    "Paris": {"winter": 4, "spring": 12, "summer": 20, "autumn": 13},
    "Tokyo": {"winter": 6, "spring": 15, "summer": 27, "autumn": 18},
    "Moscow": {"winter": -10, "spring": 5, "summer": 18, "autumn": 8},
    "Sydney": {"winter": 12, "spring": 18, "summer": 25, "autumn": 20},
    "Berlin": {"winter": 0, "spring": 10, "summer": 20, "autumn": 11},
    "Beijing": {"winter": -2, "spring": 13, "summer": 27, "autumn": 16},
    "Rio de Janeiro": {"winter": 20, "spring": 25, "summer": 30, "autumn": 25},
    "Dubai": {"winter": 20, "spring": 30, "summer": 40, "autumn": 30},
    "Los Angeles": {"winter": 15, "spring": 18, "summer": 25, "autumn": 20},
    "Singapore": {"winter": 27, "spring": 28, "summer": 28, "autumn": 27},
    "Mumbai": {"winter": 25, "spring": 30, "summer": 35, "autumn": 30},
    "Cairo": {"winter": 15, "spring": 25, "summer": 35, "autumn": 25},
    "Mexico City": {"winter": 12, "spring": 18, "summer": 20, "autumn": 15},
}

# Сопоставление месяцев с сезонами
month_to_season = {12: "winter", 1: "winter", 2: "winter",
                   3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "autumn", 10: "autumn", 11: "autumn"}

# Генерация данных о температуре
def generate_realistic_temperature_data(cities, num_years=10):
    dates = pd.date_range(start="2010-01-01", periods=365 * num_years, freq="D")
    data = []

    for city in cities:
        for date in dates:
            season = month_to_season[date.month]
            mean_temp = seasonal_temperatures[city][season]
            # Добавляем случайное отклонение
            temperature = np.random.normal(loc=mean_temp, scale=5)
            data.append({"city": city, "timestamp": date, "temperature": temperature})

    df = pd.DataFrame(data)
    df['season'] = df['timestamp'].dt.month.map(lambda x: month_to_season[x])
    return df

# Функция для вычисления статистики по сезонам
def compute_season_stats(df):
    grouped = df.groupby(["city", "season"])["temperature"]
    stats = grouped.agg(["mean", "std"]).reset_index()
    return stats

# Функция для обнаружения аномалий
def detect_anomalies(row, stats):
    subset = stats[(stats["city"] == row["city"]) & (stats["season"] == row["season"])]
    mean_val = subset["mean"].values[0]
    std_val = subset["std"].values[0]
    return abs(row["temperature"] - mean_val) > 2 * std_val

# Основная функция приложения
def main():
    st.title("Temperature Analysis & Monitoring")

    # Загрузка файла CSV
    uploaded_file = st.file_uploader("Upload temperature_data.csv", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)

        # Скользящее среднее (30-дневное окно)
        df["rolling_mean"] = df.groupby("city")["temperature"].apply(
            lambda x: x.rolling(30, min_periods=1).mean()
        )

        # Параллельное вычисление сезонных средних и стандартных отклонений
        cities_seasons = df[["city", "season"]].drop_duplicates().values.tolist()
        def process_city_season(cs):
            c, s = cs
            subset = df[(df["city"] == c) & (df["season"] == s)]
            return (c, s, subset["temperature"].mean(), subset["temperature"].std())
        results = Parallel(n_jobs=-1)(delayed(process_city_season)(cs) for cs in cities_seasons)
        stats_df = pd.DataFrame(results, columns=["city", "season", "mean", "std"])

        # Обнаружение аномалий
        df["is_anomaly"] = df.apply(lambda row: detect_anomalies(row, stats_df), axis=1)

        # Показ описательной статистики
        st.subheader("Descriptive Statistics")
        st.write(df[["city", "season", "temperature"]].describe())

        # Выбор города
        city_list = df["city"].unique().tolist()
        city = st.selectbox("Select a City", city_list)

        # Построение временного ряда температуры с аномалиями
        city_data = df[df["city"] == city]
        fig = px.line(city_data, x="timestamp", y="temperature", color=city_data["is_anomaly"].astype(str),
                      title=f"Temperature Time Series ({city})")
        st.plotly_chart(fig)

        # Секция API OpenWeatherMap
        st.subheader("Current Temperature via OpenWeatherMap (optional)")
        api_key = st.text_input("Enter your OpenWeatherMap API Key", type="password")
        if api_key:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}"
            res = requests.get(url)
            if res.status_code == 200:
                current_temp = res.json()["main"]["temp"]
                st.write(f"Current temperature in {city}: {current_temp} °C")
                # Сравнение с историческими данными (приблизительно по текущему месяцу)
                current_month = pd.to_datetime("now").month
                st.write("Comparison with historical data is approximate. Adjust logic for precise season detection.")
            else:
                st.write(res.json())

if __name__ == "__main__":
    main()