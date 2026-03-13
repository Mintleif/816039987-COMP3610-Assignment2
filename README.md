# NYC Taxi Tip Prediction

## Project Overview

This project analyzes New York City taxi trip data to understand tipping behavior and build predictive models. The goal is to use machine learning techniques to predict tipping patterns based on trip characteristics.

Two prediction tasks are explored:

- **Regression:** Predict the **tip amount** for a trip.
- **Classification:** Predict whether a trip will receive a **high tip** (defined as a tip greater than **20% of the fare**).

Multiple machine learning models are evaluated and compared, including both **traditional machine learning algorithms** and a **neural network**, in order to assess their effectiveness on large tabular datasets.

---

## Dataset

The dataset is based on **NYC Taxi Trip Records**, which include information such as:

- Pickup and dropoff timestamps  
- Trip distance  
- Fare amount  
- Tip amount  
- Passenger count  
- Payment type  
- Pickup and dropoff location IDs  

Only **credit card trips** are used because tip amounts are reliably recorded for those transactions.

Geographic context is added using the **Taxi Zone Lookup dataset**, which maps location IDs to NYC boroughs.

The dataset contains **over 1.6 million taxi trips**, making it suitable for large-scale machine learning analysis.

---

## Feature Engineering

Several engineered features were created to improve model performance.

### Temporal Features

- Pickup hour  
- Day of week  
- Weekend indicator  

These features capture time-based patterns in taxi demand and tipping behavior.

### Trip Characteristics

- Trip duration (minutes)  
- Trip speed (mph)  
- Log-transformed trip distance  

These variables describe the structure and efficiency of each trip.

### Fare-Based Features

- Fare per mile  
- Fare per minute  

These features capture the economic scale and efficiency of the trip.

### Location Features

- Pickup borough  
- Dropoff borough  

Location features were encoded using one-hot encoding to capture geographic variation in tipping behavior.

---

## Models Implemented

### Regression Models

- Linear Regression  
- Random Forest Regressor  

### Classification Models

- Logistic Regression  
- Random Forest Classifier  
- Neural Network (PyTorch)

Randomized hyperparameter tuning was applied to the **Random Forest classifier** using cross-validation.

---

## Results

### Classification Performance (Test Set)

| Model | Accuracy | F1 Score | ROC-AUC |
|------|------|------|------|
| Logistic Regression | 0.766 | 0.866 | 0.597 |
| Random Forest (Baseline) | 0.770 | 0.868 | 0.619 |
| Random Forest (Tuned) | 0.770 | 0.868 | 0.617 |
| **Neural Network** | **0.770** | **0.869** | **0.617** |

All models produced **very similar performance**, with the neural network achieving the highest F1 score by a small margin.

Earlier experimentation included the **total trip amount** feature, which produced unrealistically high performance because it leaked information about the final tip value. This feature was removed to ensure a fair and realistic prediction task.

---

### Regression Performance (Test Set)

| Model | MAE | RMSE | R² |
|------|------|------|------|
| Linear Regression | 1.249 | 2.297 | 0.609 |
| **Random Forest Regressor** | **1.191** | **2.180** | **0.648** |

The Random Forest Regressor achieved the lowest prediction error and highest explained variance.

---

## Installation

Install the required dependencies:

pip install -r requirements.txt


---

## Running the Project

Open the notebook and change cleaned dataset location. 

Now run all cells.


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

