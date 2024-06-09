from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import traceback

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('soil_data.xlsx - Bhuvana.csv')

# Impute null values with mean
data.fillna(data.mean(), inplace=True)

# Extract X (spectral bands) and y (soil properties)
X = data.iloc[:, :18]
y = data.iloc[:, 18:]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IRIV coupled with SCA
def iriv_sca(X, y, threshold=0.01, max_iter=10):
    selected_features = np.arange(X.shape[1])
    for _ in range(max_iter):
        correlations = np.array([spearmanr(X.iloc[:, i], y.iloc[:, j])[0] 
                                 for i in selected_features for j in range(y.shape[1])])
        correlations = correlations.reshape(len(selected_features), y.shape[1])
        avg_correlations = np.mean(np.abs(correlations), axis=1)
        if np.max(avg_correlations) < threshold:
            break
        least_informative_feature = np.argmin(avg_correlations)
        selected_features = np.delete(selected_features, least_informative_feature)
    return selected_features

# Apply IRIV-SCA
selected_features = iriv_sca(X_train, y_train, threshold=0.2)

# Subset Selection with Absolute Correlation Values Greater than 0.2
correlations = np.array([spearmanr(X_train.iloc[:, i], y_train.iloc[:, j])[0] 
                         for i in selected_features for j in range(y_train.shape[1])])
correlations = correlations.reshape(len(selected_features), y_train.shape[1])

# Select features with absolute correlation > 0.2
selected_features = selected_features[np.any(np.abs(correlations) > 0.2, axis=1)]

# Define regression models
models = {
    "PLSR": Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=2)),
                ("regressor", BayesianRidge())
            ]),
    "BRR": BayesianRidge(),
    "RR": Ridge(),
    "KRR": KernelRidge(),
    "SVMR": SVR(),
    "XGBoost": GradientBoostingRegressor(),
    "RFR": RandomForestRegressor()
}

# Rename features
X_train.columns = [f'feature{i}' for i in range(1, 19)]
X_test.columns = [f'feature{i}' for i in range(1, 19)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form['feature' + str(i)]) for i in range(1, 19)]
        X_new = pd.DataFrame([features], columns=[f'feature{i}' for i in range(1, 19)])
        X_new_selected = X_new.iloc[:, selected_features]

        predictions = {}
        metrics = {model: {"R2": {}, "RMSE": {}} for model in models}

        for model_name, model in models.items():
            y_preds = {}
            for i, column in enumerate(y.columns):
                model.fit(X_train.iloc[:, selected_features], y_train.iloc[:, i])
                y_pred = model.predict(X_new_selected)[0]
                y_preds[column] = y_pred
                
                y_test_pred = model.predict(X_test.iloc[:, selected_features])
                r2 = r2_score(y_test.iloc[:, i], y_test_pred)
                rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_test_pred))
                
                metrics[model_name]["R2"][column] = r2
                metrics[model_name]["RMSE"][column] = rmse

            predictions[model_name] = y_preds

        return jsonify({"predictions": predictions, "metrics": metrics})
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        print("Error:", error_message)
        print("Traceback:", error_traceback)
        return jsonify({"error": "An error occurred while predicting.", "message": error_message, "traceback": error_traceback})

if __name__ == '__main__':
    app.run(debug=True)
