from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('outputs/models/spam_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
