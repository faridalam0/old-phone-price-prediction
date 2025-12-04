import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder

# === Download model from Google Drive via requests ===
file_id = "1dkNGLbBBXANZcWqWaR0atvxNvCss4Iuu"
url = f"https://drive.google.com/uc?id={file_id}"
output = "old_phone.pkl"

st.write("Retrieving model from cloud storage")
try:
    response = requests.get(url)
    if response.status_code == 200:
        with open(output, "wb") as f:
            f.write(response.content)
        st.success("✅ Model downloaded successfully!")
    else:
        st.error(f"❌ Download failed with status code {response.status_code}")
        st.stop()
except Exception as e:
    st.error(f"❌ Error downloading model: {e}")
    st.stop()

# === Load model ===
with open(output, "rb") as f:
    model = pickle.load(f)
st.success("✅ Model loaded successfully!")

# === Load dataset (locally hosted or also from Drive) ===
df = pd.read_csv("dataset.csv")

brand_model_map = df.groupby('brand')['model'].unique().to_dict()
brand_list = list(brand_model_map.keys())

st.title("Old Phone Price Prediction")
st.sidebar.header("Enter Phone Details")

selected_brand = st.sidebar.selectbox("Select Brand", brand_list)
selected_model = st.sidebar.selectbox("Select Model", brand_model_map[selected_brand])
ram = st.sidebar.selectbox("RAM (GB)", sorted(df['ram_gb'].unique()))
storage = st.sidebar.selectbox("Storage (GB)", sorted(df['storage_gb'].unique()))
condition = st.sidebar.selectbox("Condition", df['condition'].unique())
battery = st.sidebar.slider("Battery health (%)", 50, 100, 80)
age = st.sidebar.slider("Age of Phone (Years)", 0, 5, 1)
original_price = st.sidebar.number_input("Original Price (INR)", 3000, 100000, 15000, step=500)

le_brand = LabelEncoder()
le_model = LabelEncoder()
le_condition = LabelEncoder()

df['brand'] = le_brand.fit_transform(df['brand'])
df['model'] = le_model.fit_transform(df['model'])
df['condition'] = le_condition.fit_transform(df['condition'])

brand_encoded = le_brand.transform([selected_brand])[0]
model_encoded = le_model.transform([selected_model])[0]
condition_encoded = le_condition.transform([condition])[0]

input_data = pd.DataFrame({
    "brand": [brand_encoded],
    "model": [model_encoded],
    "ram_gb": [ram],
    "storage_gb": [storage],
    "condition": [condition_encoded],
    "battery_health": [battery],
    "age_years": [age],
    "original_price": [original_price]
})

if st.sidebar.button("Predict"):
    predicted_price = model.predict(input_data)[0]
    st.success(f"Estimated Old Phone Price: ₹{int(predicted_price):,}")
