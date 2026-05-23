import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd

# -----------------------------
# Load model and preprocessors
# -----------------------------
model = tf.keras.models.load_model(
    'churn_model.h5',
    compile=False
)

with open('preprocessor_geography.pkl', 'rb') as f:
    label_encoder_geo = pickle.load(f)

with open('preprocessor_gender.pkl', 'rb') as f:
    label_encoder_gen = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    scaler = pickle.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Customer Churn Prediction")

st.write(
    "Enter customer details to predict whether "
    "the customer is likely to churn."
)

# -----------------------------
# User Inputs
# -----------------------------
credit_score = st.slider(
    "Credit Score",
    300,
    850,
    600
)

geography = st.selectbox(
    "Geography",
    label_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gen.classes_
)

age = st.slider(
    "Age",
    18,
    100,
    40
)

tenure = st.slider(
    "Tenure",
    0,
    10,
    3
)

balance = st.number_input(
    "Balance",
    value=60000.0
)

num_of_products = st.slider(
    "Number of Products",
    1,
    4,
    2
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    value=50000.0
)

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# -----------------------------
# Geography Encoding
# -----------------------------
geo_encoded = label_encoder_geo.transform(
    input_df[['Geography']]
)

if hasattr(geo_encoded, "toarray"):
    geo_encoded = geo_encoded.toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=label_encoder_geo.get_feature_names_out(
        ['Geography']
    )
)

# -----------------------------
# Gender Encoding
# -----------------------------
input_df['Gender'] = label_encoder_gen.transform(
    input_df['Gender']
)

# -----------------------------
# Drop Original Geography
# -----------------------------
input_df = input_df.drop(
    'Geography',
    axis=1
)

# -----------------------------
# Concatenate Encoded Features
# -----------------------------
final_input = pd.concat(
    [
        input_df.reset_index(drop=True),
        geo_encoded_df.reset_index(drop=True)
    ],
    axis=1
)

# -----------------------------
# Match Training Column Order
# -----------------------------
final_input = final_input[
    scaler.feature_names_in_
]

# -----------------------------
# Scale Features
# -----------------------------
final_input_scaled = scaler.transform(
    final_input
)

# -----------------------------
# Predict
# -----------------------------
prediction = model.predict(
    final_input_scaled
)

probability = prediction[0][0]

# -----------------------------
# Output
# -----------------------------
st.subheader(
    f"Churn Probability: {probability:.2f}"
)

if probability > 0.5:
    st.error(
        "The customer is likely to churn."
    )
else:
    st.success(
        "The customer is unlikely to churn."
    )