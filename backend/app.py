
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask application
superkart_api = Flask("superkart_sales_api")
CORS(superkart_api)

# Load the trained model pipeline
model = joblib.load("superkart_sales_forecast_model_v1_0.joblib")

REQUIRED_FIELDS = [
    'Product_Weight',
    'Product_Sugar_Content',
    'Product_Type_Category',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
]


def build_input_row(data):
    """Converts raw request fields into the exact columns the pipeline was trained on."""
    return {
        'Product_Weight': float(data['Product_Weight']),
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Type_Category': data['Product_Type_Category'],
        'Product_Allocated_Area_Log': np.log1p(float(data['Product_Allocated_Area'])),
        'Product_MRP': float(data['Product_MRP']),
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type']
    }


@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API"


@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        data = request.get_json()

        missing_fields = [f for f in REQUIRED_FIELDS if f not in data]
        if missing_fields:
            return jsonify({'error': f"Missing fields: {missing_fields}"}), 400

        input_df = pd.DataFrame([build_input_row(data)])
        prediction = model.predict(input_df)[0]

        return jsonify({'Predicted_Sales': round(float(prediction), 2)})

    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500


@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    try:
        file = request.files['file']
        batch_df = pd.read_csv(file)

        missing_fields = [f for f in REQUIRED_FIELDS if f not in batch_df.columns]
        if missing_fields:
            return jsonify({'error': f"Missing columns: {missing_fields}"}), 400

        rows = [build_input_row(row) for row in batch_df[REQUIRED_FIELDS].to_dict(orient='records')]
        input_df = pd.DataFrame(rows)

        predictions = model.predict(input_df).tolist()
        predictions = [round(float(p), 2) for p in predictions]

        output_dict = dict(zip(batch_df.index.astype(str), predictions))
        return jsonify(output_dict)

    except Exception as e:
        return jsonify({'error': f"Batch prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    superkart_api.run(debug=True)
