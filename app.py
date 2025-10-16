import streamlit as st
import pickle
import pandas as pd
import gdown
from sklearn.preprocessing import LabelEncoder

# === Download model from Google Drive ===
FILE_ID = "1dkNGLbBBXANZcWqWaR0atvxNvCss4Iuu"
url = f"https://drive.google.com/uc?id={FILE_ID}"
output = "old_phone.pkl"
gdown.download(url, output, quiet=True)

# === Load model ===
with open(output, 'rb') as f:
    model = pickle.load(f)

# === Load dataset (upload dataset.csv in same folder or also from Drive) ===
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
original_price = st.sidebar.number_input("Original Price (INR)", 3000, 100000, 15000)

# Label Encoding
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

