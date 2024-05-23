import pickle
from flask import Flask, render_template, request
# from flask_ngrok import run_with_ngrok

with open('housePricePridictor.pkl','rb') as file:
  lm=pickle.load(file)

# prediction=lm.predict([[-0.915249,0.877002,-0.522275]])
# print(prediction)


app = Flask(__name__)

# run_with_ngrok(app)

@app.route("/")
def index():
    return render_template("index.html")  # Renders the HTML template 

@app.route("/predict", methods=["POST"])
def predict():
    # Get user input from the form
    area_income = request.form.get("avg_area_income")
    avg_area_house_age = request.form.get("avg_area_house_age")
    avg_area_rooms = request.form.get("avg_area_rooms")

    # Convert input to a list of floats
    user_input = [
        float(area_income),
        float(avg_area_house_age),
        float(avg_area_rooms),
    ]

    # Make prediction using your model
    prediction = lm.predict([user_input])

    # Round the prediction to 2 decimal places
    predicted_price = round(prediction[0], 2)

    return render_template("predict.html", predicted_price=predicted_price)

# if __name__ == "__main__":
app.run(debug=True)

