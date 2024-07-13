# Red Wine Quality Prediction
This repository contains the code and instructions for building a machine learning model to predict the quality of red wine and deploying it as a web application using Flask.
![app image](https://github.com/user-attachments/assets/64429f5d-a7af-4385-9d5e-4a13b2c95c7d)
# Table of Contents
- Introduction
- Prerequisites
- [Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
- Model Training
- Flask Web Application
- Running the Application

# Introduction
This project demonstrates how to:
1. Download and preprocess the red wine quality dataset from Kaggle.
2. Train a machine learning model to predict wine quality.
3. Deploy the trained model as a web application using Flask.

# Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.x
- Anaconda
- Flask
- Scikit-learn
- Pandas
- Pickle

# Dataset
![dataset](https://github.com/user-attachments/assets/89cf2e6d-85d2-4d9d-bb43-6c1e38b972a9)
1. Download the red wine quality dataset from Kaggle: [Red Wine Quality Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009).
2. Extract the dataset to your project directory.



# Model Training
1. Open a Jupyter notebook or a Python script.
2. Load the dataset and preprocess it.
3. Train a machine learning model.
4. Save the trained model using pickle.

##### [N.B] For this part You can run the "Wine_Quality_Prediction.ipynb" file from the repository.




# Making Flask Web Application:

### 1. Create a file named app.py and add the following code to it
```python
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, static_url_path='/static')

# Load the saved model
model = pickle.load(open('wine_model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
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

```

### 2. Create an index.html file in a templates folder with the following code:

```<!DOCTYPE html>
<html>
<head>
    <title>Red Wine Quality Prediction</title>
    <style>
        body {
            background-image: url("{{ url_for('static', filename='red-wine.jpg') }}");
            background-size: cover;
            font-family: Arial, sans-serif;
        }
        form {
            text-align: center;
        }
        input {
            width: 50%;
        }
        input[type=submit] {
            background-color: green;
            color: white;
            width: 10%;
        }
        select {
            width: 50%;
        }
        label {
            font-size: 20px;
            display: inline-block;
            width: 150px;
            text-align: right;
        }
    </style>
</head>
<body>
    <center><h1>Red Wine Quality Prediction: {{ results }}</h1></center>
    <form action="{{ url_for('predict') }}" method="post">
        <label for="fixed_acidity">Fixed Acidity</label>
        <input type="text" id="fixed_acidity" name="fixed_acidity" placeholder="Enter fixed acidity..." /><br><br>
        <label for="volatile_acidity">Volatile Acidity</label>
        <input type="text" id="volatile_acidity" name="volatile_acidity" placeholder="Enter volatile acidity..." /><br><br>
        <label for="citric_acid">Citric Acid</label>
        <input type="text" id="citric_acid" name="citric_acid" placeholder="Enter citric acid..." /><br><br>
        <label for="residual_sugar">Residual Sugar</label>
        <input type="text" id="residual_sugar" name="residual_sugar" placeholder="Enter residual sugar..." /><br><br>
        <label for="chlorides">Chlorides</label>
        <input type="text" id="chlorides" name="chlorides" placeholder="Enter chlorides..." /><br><br>
        <label for="free_sulfur_dioxide">Free Sulfur Dioxide</label>
        <input type="text" id="free_sulfur_dioxide" name="free_sulfur_dioxide" placeholder="Enter free sulfur dioxide..." /><br><br>
        <label for="total_sulfur_dioxide">Total Sulfur Dioxide</label>
        <input type="text" id="total_sulfur_dioxide" name="total_sulfur_dioxide" placeholder="Enter total sulfur dioxide..." /><br><br>
        <label for="density">Density</label>
        <input type="text" id="density" name="density" placeholder="Enter density..." /><br><br>
        <label for="pH">pH</label>
        <input type="text" id="pH" name="pH" placeholder="Enter pH..." /><br><br>
        <label for="sulphates">Sulphates</label>
        <input type="text" id="sulphates" name="sulphates" placeholder="Enter sulphates..." /><br><br>
        <label for="alcohol">Alcohol</label>
        <input type="text" id="alcohol" name="alcohol" placeholder="Enter alcohol..." /><br><br>
        <input id="submit" type="submit" value="Predict" />
    </form>
</body>
</html>
```

# Running the Application
1. Open Anaconda Prompt.
2. Navigate to your project directory using the cd command.

```
cd/d path/to/your/project/directory

```

3. Then type and Enter
```
python app.py
```
4. Now you will see an web address like this.


![address](https://github.com/user-attachments/assets/1856f37d-254a-49cb-8c69-5292765f6b5c)

5. Copy the address and Open a web browser and navigate to "http://127.0.0.1:5000/"  

(This won't run in your system.This is for example)


### Author: H.M. Nahid kawsar
Find me in [LinkedIn:](#linkedin.com/in/h-m-nahid-kawsar-232a86266)
