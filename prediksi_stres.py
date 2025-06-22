import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Data dummy
data = {
    'sleep_hours': [8, 7, 5, 6, 4, 9, 3],
    'exercise_mins': [30, 45, 10, 15, 0, 60, 5],
    'caffeine_cups': [1, 2, 3, 4, 5, 0, 6],
    'stress_level': ['Rendah', 'Rendah', 'Sedang', 'Sedang', 'Tinggi', 'Rendah', 'Tinggi']
}
df = pd.DataFrame(data)

# Model
X = df[['sleep_hours', 'exercise_mins', 'caffeine_cups']]
y = df['stress_level']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit App
st.set_page_config(page_title="Deteksi Stres", layout="centered")
st.title("🧘 Prediksi Tingkat Stres")

sleep = st.slider("🛌 Jam Tidur", 0, 12, 7)
exercise = st.slider("🏃‍♀️ Olahraga (menit)", 0, 120, 30)
caffeine = st.slider("☕ Kafein (cangkir)", 0, 10, 2)

input_data = pd.DataFrame([[sleep, exercise, caffeine]], columns=X.columns)
hasil = model.predict(input_data)[0]
st.success(f"🧠 Prediksi Stres Kamu: **{hasil}**")
