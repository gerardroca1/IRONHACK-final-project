from flask import Flask, request, render_template
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = request.form.to_dict()
        logging.debug(f"Form data received: {data}")

        # Convert form data to floats and handle comma to period conversion
        data = {k: float(v.replace(',', '.')) for k, v in data.items()}
        df = pd.DataFrame([data])

        # Feature Engineering
        df['sulphates_alcohol_interaction'] = df['sulphates'] * df['alcohol']

        # Normalize the numerical features
        scaled_features = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled_features, columns=df.columns)

        # Make prediction
        prediction = model.predict(scaled_df)

        # Render the result on the same page
        return render_template('index.html', prediction=int(prediction[0]))
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
