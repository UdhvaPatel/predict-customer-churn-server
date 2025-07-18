# 🔧 Customer Churn Prediction – Backend API (Flask)

This is the **Flask-based backend** for the Customer Churn Prediction web app. It hosts a trained machine learning model (Logistic Regression) that predicts the likelihood of a customer churning based on input features.

---

## 🚀 Live API

🌐 [Backend API on Render](https://predict-customer-churn-server.onrender.com/predict)

Use a `POST` request to the `/predict` endpoint to receive a churn prediction and probability.

---

## 📥 Example Request Payload

```json
{
  "tenure": 24,
  "MonthlyCharges": 65.5,
  "TotalCharges": 1572.0,
  "Contract": "Month-to-month",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "TechSupport": "No",
  "PaperlessBilling": "Yes",
  "Partner": "No",
  "PaymentMethod": "Electronic check",
  "gender": "Male",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "Dependents": "No",
  "SeniorCitizen": 0
}
```
## 📤 Example Response
```
{
  "churn": 1,
  "probability": 0.87
}
```
## 🧠 Model Details
Algorithm: Logistic Regression

Libraries: scikit-learn, pandas, joblib

Preprocessing: Categorical encoding using pd.get_dummies

Serialization: Model and column transformer saved using joblib


## 🛠️ Tech Stack
Python 3

Flask

scikit-learn

pandas

joblib

flask-cors
## 📦 Setup Instructions (Local)
Clone the project and navigate to the server folder:
```
cd predict-customer-churn-server
```

Install dependencies:
```
pip install -r requirements.txt
```

Run the Flask app:
```
python app.py
```

Test endpoint:
Run with Postman or curl to ```http://localhost:5000/predict```

## 🔗 Connects To Frontend
This backend powers the React frontend at:

🔗 [Frontend on Vercel](https://predict-customer-churn.vercel.app/)
 ## Project Structure
```
 server/
├── app.py                 # Flask app with /predict endpoint
├── churn_model.pkl        # Trained ML model
├── model_columns.pkl      # Feature columns used during training
├── requirements.txt       # Python dependencies
```
## Author 
Udhva Patel
