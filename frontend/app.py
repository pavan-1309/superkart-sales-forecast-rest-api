
import streamlit as st
import requests

# Base URL of the Flask backend (Docker network alias)
BACKEND_URL = "http://backend:7860"

st.title("🛒 SuperKart Sales Forecasting App")
st.markdown("Enter product and store attributes to forecast **product sales revenue**.")

# --- Online Prediction ---
st.subheader("Online Prediction")

Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Type = st.selectbox("Product Type", [
    "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
    "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
    "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"
])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.1)
Product_MRP = st.number_input("Maximum Retail Price", min_value=0.0, value=150.0)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])

product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Type": Product_Type,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type
}

if st.button("Predict", type='primary'):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data)
        if response.status_code == 200:
            predicted_sales = response.json()["Predicted_Sales"]
            st.success(f"Predicted Sales: **${predicted_sales:,.2f}**")
        else:
            st.error(f"API Error: {response.json().get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Connection error: {e}")

# --- Batch Prediction ---
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file})
        if response.status_code == 200:
            st.success("Batch predictions completed!")
            st.write(response.json())
        else:
            st.error(f"API Error: {response.json().get('error', 'Unknown error')}")
