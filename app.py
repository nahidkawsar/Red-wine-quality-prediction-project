from flask import Flask, request, render_template
import pickle

app = Flask(__name__, static_url_path='/static')

model = pickle.load(open('wine_model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])

            # Create a feature list for the model
            features = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                         pH, sulphates, alcohol]]

            # Predict the wine quality
            prediction = model.predict(features)
            pred = prediction[0]
            out = "Good Quality Wine" if pred == 1 else "Bad Quality Wine"

            return render_template('index.html', results=out)
        except Exception as e:
            print(f"An error occurred: {e}")
            return render_template('index.html', results="Error in prediction")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
