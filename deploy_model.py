import streamlit as st
import joblib
import numpy as np

# Load your models
loaded_xgb = joblib.load('xgb_model.joblib')
loaded_dt = joblib.load('decision_tree_model.joblib')

#load the scalers
min_max_scaler = joblib.load('min_max_scaler.joblib')
std_scaler = joblib.load('std_scaler.joblib')

# Label mapping
label_mapping = {
    0: 'At-Risk Customers',
    1: 'Best Customers',
    2: 'Churned Best Customers',
    3: 'Lost Customers',
    4: 'Lowest-Spending Active Loyal Customers',
    5: 'High-Spending New Customers',

}

# Function to perform prediction based on selected model and input data
def predict(model, input_data):
    # Transformasi input_data sesuai dengan skala yang sudah ditentukan
    input_data_normalized = min_max_scaler.transform(input_data)  # Normalisasi dengan MinMaxScaler
    input_data_standardized = std_scaler.transform(input_data_normalized)  # Standarisasi dengan StandardScaler
    prediction = model.predict(input_data_standardized)
    predicted_label = label_mapping.get(prediction[0], 'Others')  # Default to 'Others' if not found
    return predicted_label

# Streamlit interface
st.title('Customer Segmentation')

model_choice = st.selectbox('Select Model', ('XGBoost', 'Decision Tree'))

gender = st.select_slider('Gender', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
refund = st.number_input('Refund', min_value=0, max_value=9999999, value=0)
wallet_balance = st.number_input('Wallet Balance', min_value=0, max_value=9999999, value=0)

# List of products
products = [
    'None', 'Man Fashion', 'Woman Fashion', 'Food & Drink', 'Ride Hailing',
    'Keperluan Rumah Tangga', 'Travel', 'Keperluan Anak', 'Elektronik', 'Other',
    'Transportasi (Kereta Pesawat Kapal)', 'Top Up Game', 'Otomotif', 'Pulsa',
    'Kesehatan', 'Investasi', 'Sewa Motor/Mobil', 'Hotel', 'Tagihan (WIFI PLN)'
]

# Function to format product names
def product_format_func(x):
    return products[x]

# Slider for most bought product
most_bought_product = st.selectbox(
    'Most Bought Product',
    options=list(range(len(products))),
    format_func=product_format_func
)

st.write(f'You selected: {products[most_bought_product]}')

total_gross_amount = st.number_input('Total Gross Amount', min_value=0, max_value=9999999, value=0)
total_discount_amount = st.number_input('Total Discount Amount', min_value=0, max_value=9999999, value=0)
recency = st.number_input('Recency', min_value=0, max_value=1000, value=0)
frequency = st.number_input('Frequency', min_value=0, max_value=1000, value=0)
monetary = st.number_input('Monetary', min_value=0, max_value=9999999, value=0)

# Prepare input data for prediction
input_data = np.array([[gender, refund, wallet_balance, most_bought_product,
                        total_gross_amount, total_discount_amount, recency, frequency, monetary]])

# Normalisasi dan standarisasi input_data sesuai dengan skala yang sudah ditentukan
input_data_normalized = min_max_scaler.transform(input_data)  # Normalisasi dengan MinMaxScaler
input_data_standardized = std_scaler.transform(input_data_normalized)  # Standarisasi dengan StandardScaler


# Perform prediction on button click
if st.button('Predict'):
    if model_choice == 'XGBoost':
        result = predict(loaded_xgb, input_data_standardized)
    elif model_choice == 'Decision Tree':
        result = predict(loaded_dt, input_data_standardized)

    st.write(f'Prediction: {result}')
