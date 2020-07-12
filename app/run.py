# Import required libraries
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# load preprocessor pipeline and model
preprocessor = joblib.load('../models/preprocessor.pkl')
model = joblib.load('../models/model.pkl')


# web page that handles user's query and displays model results
@app.route('/')
def home():
    return render_template('go.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        text = [message]
        test = preprocessor.transform(text)
        prediction = model.predict(test)
        return render_template('master.html', prediction=prediction)


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()
