import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('random_over_sampling_model.pkl')

# Function to create the new columns from original ones
def preprocess_input(data):
    # Using logical conditions to handle values
    data['IMAGES_AND_REVIEWS'] = ((data['IMAGES'] > 0) & (data['REVIEWS'] > 0)).astype(int)
    data['SPECS_AND_REVIEWS'] = ((data['SPECS'] > 0) & (data['REVIEWS'] > 0)).astype(int)
    data['FAQ_AND_IMAGES'] = ((data['FAQ'] > 0) & (data['IMAGES'] > 0)).astype(int)
    data['WARRANTY_AND_SPECS'] = ((data['WARRANTY'] > 0) & (data['SPECS'] > 0)).astype(int)
    data['COMPARE_SIMILAR_AND_SPONSORED_LINKS'] = ((data['COMPARE_SIMILAR'] > 0) & (data['SPONSORED_LINKS'] > 0)).astype(int)
    return data

# Streamlit UI
st.title("Purchase Prediction Model")

# User input section
st.write("Please input the following details (0 or 1):")

# Collect input data for all original features (values are limited to 0 or 1)
IMAGES = st.radio('IMAGES', [0, 1])
REVIEWS = st.radio('REVIEWS', [0, 1])
FAQ = st.radio('FAQ', [0, 1])
SPECS = st.radio('SPECS', [0, 1])
SHIPPING = st.radio('SHIPPING', [0, 1])
BRO_TOGETHER = st.radio('BRO_TOGETHER', [0, 1])
COMPARE_SIMILAR = st.radio('COMPARE_SIMILAR', [0, 1])
VIEW_SIMILAR = st.radio('VIEW_SIMILAR', [0, 1])
WARRANTY = st.radio('WARRANTY', [0, 1])
SPONSORED_LINKS = st.radio('SPONSORED_LINKS', [0, 1])

# Create a dataframe with user input
input_data = pd.DataFrame({
    'IMAGES': [IMAGES],
    'REVIEWS': [REVIEWS],
    'FAQ': [FAQ],
    'SPECS': [SPECS],
    'SHIPPING': [SHIPPING],
    'BRO_TOGETHER': [BRO_TOGETHER],
    'COMPARE_SIMILAR': [COMPARE_SIMILAR],
    'VIEW_SIMILAR': [VIEW_SIMILAR],
    'WARRANTY': [WARRANTY],
    'SPONSORED_LINKS': [SPONSORED_LINKS],
})

# Preprocess the input to add the new columns
processed_data = preprocess_input(input_data)

# Display the processed data (for transparency)
st.write("Processed Data for Prediction:")
st.write(processed_data)

# Make prediction using the loaded model
if st.button("Predict"):
    prediction = model.predict(processed_data)
    
    # Display the prediction result
    if prediction[0] == 1:
        st.success("The model predicts: BUY")
    else:
        st.warning("The model predicts: NOT BUY")
