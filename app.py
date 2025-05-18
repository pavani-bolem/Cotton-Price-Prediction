from flask import Flask, render_template, request
import pandas as pd
import pickle
import datetime
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load model, scaler, and encoders
with open('cotton_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load dataset
df = pd.read_csv('processed_cotton_prices.csv')
df['arrival_date'] = pd.to_datetime(df['arrival_date'])


@app.route('/')
def home():
    states = df['state'].unique().tolist()
    return render_template('index.html', states=states)


@app.route('/get_districts', methods=['POST'])
def get_districts():
    state = request.form['state']
    districts = df[df['state'] == state]['district'].unique().tolist()
    return {"districts": districts}


@app.route('/get_markets', methods=['POST'])
def get_markets():
    state = request.form['state']
    district = request.form['district']
    markets = df[(df['state'] == state) & (df['district'] == district)]['market'].unique().tolist()
    return {"markets": markets}


@app.route('/predict', methods=['POST'])
def predict():
    state = request.form['state']
    district = request.form['district']
    market = request.form['market']
    date_str = request.form['date']

    try:
        # Convert input date to numerical format
        input_date = pd.to_datetime(date_str)
        days_since_start = (input_date - df['arrival_date'].min()).days

        # Encode categorical inputs
        state_encoded = label_encoders['state'].transform([state])[0]
        district_encoded = label_encoders['district'].transform([district])[0]
        market_encoded = label_encoders['market'].transform([market])[0]

        # Prepare input data
        input_data = scaler.transform([[state_encoded, district_encoded, market_encoded, days_since_start]])

        # Predict price
        predicted_price = model.predict(input_data)[0]
        result = f"Predicted Cotton Price in {market}, {district}, {state}, {date_str}: ₹{predicted_price:.2f}"
    except:
        result = "Invalid input. Please try again."

    return render_template('index.html', states=df['state'].unique().tolist(), prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
