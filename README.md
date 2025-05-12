# ElderSafe  
### Fall Risk Prediction System for Elderly Care

## Project Objective
- Predict fall risk in elderly individuals using health and lifestyle features.
- Provide interpretable and actionable outputs for caregivers and individuals.

## Model and Tools
- **Model**: XGBoost Classifier
- **Optimization**: Optuna
- **Preprocessing**: StandardScaler
- **Language**: Python

## Input Features
- **Age** (in years)
- **Walking Speed** (in meters per second)
- **Missed Meals** (number of meals skipped in a week)
- **Sleep Hours** (average hours of sleep per day)
- **Medication Adherence** (1 = Good, 0 = Poor)
- **Past Falls** (number of falls in the past)
- **Mobility Aid** (1 = Uses aid, 0 = Doesn’t use aid)

## Model Used

- **XGBoost Classifier**
- Optimized using **Optuna**
- Standardized with **StandardScaler**

## Files Included
- `elder_safe_fall_risk_model.joblib` – Trained XGBoost model.
- `scaler.joblib` – Fitted StandardScaler object.
- `predict_fall_risk.py` – Python script to take user input and return prediction.
- `ElderSafe.ipynb` – Jupyter notebook for development and training.

## How It Works
1. **User Input**: The user enters details when prompted in the console.
2. **Data Formatting**: Inputs are formatted into a pandas DataFrame.
3. **Scaling**: Data is scaled using the pre-fitted StandardScaler.
4. **Prediction**: Model predicts fall risk category (High/Low).
5. **Output**: Prediction shown as "High Risk" or "Low Risk".

## Sample Run
### Input:
- **Age**: 78
- **Walking Speed**: 0.6 m/s
- **Missed Meals**: 3
- **Sleep Hours**: 6.5
- **Medication Adherence**: 1
- **Past Falls**: 2
- **Mobility Aid**: 1

### Output:
- **Prediction**: High Risk of Fall

## Usage
1. Clone the repository or download the files locally.
2. Install required packages:
   ```bash
   pip install pandas joblib xgboost scikit-learn

   ```
3. Run the script:
   ```bash
   python predict_fall_risk.py
   ```
4. Enter the inputs when prompted.

## Limitations

- The model is only as good as the dataset it was trained on.
- It does not cover rare medical conditions or external factors (e.g., slippery floors, poor lighting).
- Currently uses console-based inputs; not ideal for non-technical users.

## Future Advancements

- Develop a **mobile app or web interface** for easier access.
- Add **voice-based input** for seniors with limited mobility.
- Integrate real-time sensor data (e.g., smartwatches, fitness trackers).
- Retrain model with larger and more diverse datasets to improve accuracy.
- Include explainability features like **SHAP plots**.

##  Status

 Completed core ML system  
 App development postponed (will be added later)

---

###  Author

 Barun Saha 
 [LinkedIn](https://linkedin.com/in/barunsaha03/)  | [GitHub](https://github.com/Barun-LmBkr) 
