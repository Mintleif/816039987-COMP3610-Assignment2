# NYC Taxi Tip Prediction

## Project Overview
This project analyzes New York City taxi trip data to understand tipping behavior and build predictive models. The goal is to use machine learning techniques to predict tipping patterns based on trip characteristics.

Two prediction tasks are explored:

- **Regression:** Predict the tip amount for a trip.
- **Classification:** Predict whether a trip will receive a **high tip** (defined as a tip greater than 20% of the fare).

The project compares traditional machine learning models with a neural network to evaluate their effectiveness on large tabular datasets.

---

## Dataset
The dataset is based on **NYC Taxi Trip Records**, which include:

- Pickup and dropoff timestamps  
- Trip distance  
- Fare amount  
- Tip amount  
- Passenger count  
- Payment type  
- Pickup and dropoff location IDs  

Only **credit card trips** are used because tip amounts are reliably recorded for those transactions.

Geographic information is added using the **Taxi Zone Lookup dataset**, which maps location IDs to boroughs.

---

## Feature Engineering
Several features were created to improve model performance:

### Temporal Features
- Pickup hour
- Day of week
- Weekend indicator

### Trip Characteristics
- Trip duration
- Trip speed
- Log-transformed trip distance

### Fare-Based Features
- Fare per mile
- Fare per minute

### Location Features
- Pickup borough
- Dropoff borough

---

## Models Implemented

### Regression Models
- Linear Regression
- Random Forest Regressor

### Classification Models
- Logistic Regression
- Random Forest Classifier
- Neural Network (PyTorch)

Randomized hyperparameter tuning was applied to the Random Forest classifier.

---

## Results

### Classification Performance (Test Set)

| Model | Accuracy | F1 Score | ROC-AUC |
|------|------|------|------|
| Logistic Regression | 0.873 | 0.921 | 0.882 |
| Random Forest (Baseline) | 0.869 | 0.920 | 0.951 |
| Random Forest (Tuned) | 0.945 | 0.965 | 0.987 |
| **Neural Network** | **0.961** | **0.974** | **0.991** |

The neural network achieved the strongest overall classification performance.

### Regression Performance (Test Set)

| Model | MAE | RMSE | R² |
|------|------|------|------|
| Linear Regression | 0.841 | 1.380 | 0.859 |
| Random Forest Regressor | **0.679** | 1.415 | 0.852 |

The Random Forest Regressor achieved the lowest average prediction error.

---

## Installation

Install the required dependencies:
pip install -r requirements.txt


---

## Running the Project

Open the notebook and run all cells:

jupyter notebook


The notebook performs:

1. Data loading and preprocessing  
2. Feature engineering  
3. Model training and tuning  
4. Neural network training  
5. Model evaluation and interpretation  

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Matplotlib

---

## Notes
Large datasets are excluded from version control using `.gitignore`.