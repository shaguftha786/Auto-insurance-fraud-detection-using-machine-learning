from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('dtc_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        # Convert values to integers
        for key in data:
            data[key] = int(data[key])

        print(data)
        df = pd.DataFrame(data, index=[0])
        # Example prediction logic
        prediction = model.predict(df)
        
        return render_template('predict.html', predict=prediction)
    except Exception as e:
        print(f"Error: {e}")
        return "Bad Request", 400

if __name__ == '__main__':
    app.run(debug=True)
