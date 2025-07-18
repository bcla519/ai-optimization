import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy.stats import logistic
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV


# Reads dataset csv file
df = pd.read_csv("DQN1 Dataset.csv")
pd.set_option('display.max_columns', None) # show all columns

# Exploratory Data Analysis (EDA)
# Histograms for all features
# Notice month column is a single number --> month == irrelevant column
# df.select_dtypes("number").hist(bins = 20, figsize = (12,24), edgecolor = "black", layout = (10, 4))
# plt.subplots_adjust(hspace = 0.8, wspace = 0.8)
# plt.show()

# Box Plots for all features
# df.select_dtypes("number").plot(kind='box', subplots = True, figsize = (12,24), layout = (10, 4))

# Scatter Plots of all features against Health Risk Score
# Find patterns, trends, clusters, or outliers
# numerical_cols = df.select_dtypes("number").columns.values
# numerical_cols = numerical_cols[:-1]  # Get rid of Health Risk Score column
# fig, axes = plt.subplots(nrows = 9, ncols = 4, figsize = (12,24))
# plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
# current_column = 0 # track which column from numerical_cols loop is currently on
# for i in range(9):
#     for j in range(4):
#         if current_column < len(numerical_cols):
#             axes[i, j].scatter(df[numerical_cols[current_column]], df['healthRiskScore'], s=10)
#             axes[i, j].set_title(numerical_cols[current_column])
#             current_column += 1
#         else:
#             break
# plt.show()

# Heat Map for Multicollinearity
# correlation_matrix = df.corr(numeric_only=True)
# plt.figure(figsize = (20,12))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f')
# plt.show()

# Drop columns from heat map that have > 0.8 correlation
highly_correlated_columns = ['tempmax', 'tempmin', 'feelslikemax', 'feelslikemin', 'feelslike', 'windgust',
                             'solarradiation', 'solarenergy', 'sunriseEpoch', 'sunsetEpoch', 'moonphase', 'heatIndex']
df.drop(columns = highly_correlated_columns, inplace = True)

# Drop Irrelevant Data
irrelevant_cols = ['datetimeEpoch', 'month', 'dayOfWeek', 'isWeekend']
df.drop(columns = irrelevant_cols, inplace = True)

# Updated Heat Map for Multicollinearity
# correlation_matrix = df.corr(numeric_only=True)
# plt.figure(figsize = (20,12))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f')
# plt.show()

# Standardization for all features except target variable (Health Risk Score)
feature_cols = df.select_dtypes("number").columns.values
feature_cols = feature_cols[:-1] # Drop Health Risk Score column
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Split data into training and testing sets
X = df[feature_cols]
y = df['healthRiskScore']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% test, 80% training

# Creating evaluation array for different models
model_comparison_data = []
'''
# ----------- RANDOM FOREST REGRESSOR MODEL -----------
# Create and Fit to Random Forest Regressor Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Make predictions for random forest regressor
y_pred_rf = rf_model.predict(X_test)

# Evaluate random forest regressor
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, y_pred_rf)

# Add evaluation data to model_comparison_data array
model_comparison_data.append({'Model': 'Random Forest Regressor', 'MSE': rf_mse, 'RMSE': rf_rmse, 'R2': rf_r2})


# ------------------------------ TASK 2, PART B1 ------------------------------
# ---- RANDOM FOREST REGRESSOR MODEL - HYPERPARAMETER OPTIMIZATION WITH GRIDSEARCHCV ----
# Hyperparameter optimization using GridSearchCV
# GridSearchCV performs an exhaustive search through a manually specified list of parameters
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create and Fit GridSearchCV model
grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Find best Random Forest Regressor model with GridSearchCV
tuned_rf_gs = grid_search.best_estimator_

# Make predictions with tuned Random Forest Regressor model with GridSearchCV
y_pred_tuned_rf_gs = tuned_rf_gs.predict(X_test)

# Evaluate the tuned Random Forest Regressor model with GridSearchC
tuned_rf_gs_mse = mean_squared_error(y_test, y_pred_tuned_rf_gs)
tuned_rf_gs_rmse = np.sqrt(tuned_rf_gs_mse)
tuned_rf_gs_r2 = r2_score(y_test, y_pred_tuned_rf_gs)

# Add evaluation metrics to model_comparison_data array
model_comparison_data.append({'Model': 'RF - GridSearchCV', 'MSE': tuned_rf_gs_mse, 'RMSE': tuned_rf_gs_rmse, 'R2': tuned_rf_gs_r2})

# ---- RANDOM FOREST REGRESSOR MODEL - HYPERPARAMETER OPTIMIZATION WITH RANDOMIZEDSEARCHCV ----
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create and Fit RandomizedSearchCV
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1, scoring='r2')
random_search.fit(X_train, y_train)

# Find best Random Forest Regressor model with RandomizedSearchCV
tuned_rf_rs = random_search.best_estimator_

# Make predictions with tuned Random Forest Regressor model with RandomizedSearchCV
y_pred_tuned_rf_rs = tuned_rf_rs.predict(X_test)

# Evaluate the tuned Random Forest Regressor model with RandomizedSearchCV
tuned_rf_rs_mse = mean_squared_error(y_test, y_pred_tuned_rf_rs)
tuned_rf_rs_rmse = np.sqrt(tuned_rf_rs_mse)
tuned_rf_rs_r2 = r2_score(y_test, y_pred_tuned_rf_rs)

# Add evaluation metrics to model_comparison_data array
model_comparison_data.append({'Model': 'RF - RandomizedSearchCV', 'MSE': tuned_rf_rs_mse, 'RMSE': tuned_rf_rs_rmse, 'R2': tuned_rf_rs_r2})


# ------------------------------ TASK 2, PART B2 ------------------------------
# ----------- REGULARIZATION WITH RIDGE CV -----------
# Create and Fit to RidgeCV Model
ridgecv_model = RidgeCV(cv=5)
ridgecv_model.fit(X_train, y_train)

# Make predictions for RidgeCV model
y_pred_ridgecv = ridgecv_model.predict(X_test)

# Evaluate RidgeCV model
ridgecv_mse = mean_squared_error(y_test, y_pred_ridgecv)
ridgecv_rmse = np.sqrt(ridgecv_mse)
ridgecv_r2 = r2_score(y_test, y_pred_ridgecv)

# Add evaluation data to model_comparison_data array
model_comparison_data.append({'Model': 'RidgeCV', 'MSE': ridgecv_mse, 'RMSE': ridgecv_rmse, 'R2': ridgecv_r2})

# ----------- REGULARIZATION WITH ELASTIC NET CV -----------
# Create and Fit to ElasticNetCV
elastic_model = ElasticNetCV(cv=5)
elastic_model.fit(X_train, y_train)

# Make predictions for ElasticNetCV
y_pred_elastic = elastic_model.predict(X_test)

# Evaluate ElasticNetCV
elastic_mse = mean_squared_error(y_test, y_pred_elastic)
elastic_rmse = np.sqrt(elastic_mse)
elastic_r2 = r2_score(y_test, y_pred_elastic)

# Add evaluation data to model_comparison_data array
model_comparison_data.append({'Model': 'Elastic Net CV', 'MSE': elastic_mse, 'RMSE': elastic_rmse, 'R2': elastic_r2})


# ------------------------------ TASK 2, PART B3 ------------------------------
# ----------- ENSEMBLE - BAGGING REGRESSOR -----------
# Create and Fit to Bagging Regressor
bagging_model = BaggingRegressor(random_state=42)
bagging_model.fit(X_train, y_train)

# Make predictions for Bagging Regressor
y_pred_bagging = bagging_model.predict(X_test)

# Evaluate Bagging Regressor
bagging_mse = mean_squared_error(y_test, y_pred_bagging)
bagging_rmse = np.sqrt(bagging_mse)
bagging_r2 = r2_score(y_test, y_pred_bagging)

# Add evaluation data to model_comparison_data array
model_comparison_data.append({'Model': 'Bagging Regressor', 'MSE': bagging_mse, 'RMSE': bagging_rmse, 'R2': bagging_r2})
'''
# ----------- ENSEMBLE - EXTRA TREES REGRESSOR -----------
# Create and Fit to Extra Trees Regressor
et_model = ExtraTreesRegressor(random_state=42)
et_model.fit(X_train, y_train)

# Make predictions for Extra Trees Regressor
y_pred_et = et_model.predict(X_test)

# Evaluate Extra Trees Regressor
et_mse = mean_squared_error(y_test, y_pred_et)
et_rmse = np.sqrt(et_mse)
et_r2 = r2_score(y_test, y_pred_et)

# Add evaluation data to model_comparison_data array
model_comparison_data.append({'Model': 'Extra Trees Regressor', 'MSE': et_mse, 'RMSE': et_rmse, 'R2': et_r2})


# ------------------------------ TASK 2, PART C ------------------------------
# ----------- PRINT MODEL COMPARISON SUMMARY TABLE -----------
print('---- MODEL COMPARISON SUMMARY TABLE ----')
comparison_df = pd.DataFrame(model_comparison_data)
comparison_df.set_index('Model', inplace=True)

# Sort by R-squared values
comparison_df_sorted = comparison_df.sort_values('R2', ascending=False)
print(comparison_df_sorted)

# ------------------------------ SHAPLEY VALUES ------------------------------
explainer_et = shap.Explainer(et_model)
shap_values_et = explainer_et(X_train)

shap.plots.beeswarm(shap_values_et)
# shap.plots.waterfall(shap_values_et)