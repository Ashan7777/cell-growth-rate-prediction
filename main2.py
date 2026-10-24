import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# Import the dataset
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_4factor.csv"))

# Extract the feature and target columns
features = df.columns[1:]
targets = df.columns[0]

# Convert to numpy
x=df.loc[:,features].to_numpy()
y=df.loc[:,targets].to_numpy()

# Creating training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =0)

# Linear Reg
model_lin = make_pipeline(StandardScaler(),LinearRegression())

# 4fold cross validation
cv_results_lin = cross_validate(model_lin, x_train, y_train, cv=4,scoring=("neg_root_mean_squared_error", "r2"),return_train_score=True)

# Training and validation metrix
ms_train_rmse_lin = -cv_results_lin["train_neg_root_mean_squared_error"]
ms_train_r2_lin = cv_results_lin["train_r2"]

ms_valid_rmse_lin = -cv_results_lin["test_neg_root_mean_squared_error"]
ms_valid_r2_lin = cv_results_lin["test_r2"]


# Find RMSE and standard deviation of the metrics
print()
print("Linear Regression")
print(f"Train | RMSE | {np.mean(ms_train_rmse_lin):6.3f} +/- {np.std(ms_train_rmse_lin):6.3f} d^-1")
print(f"Valid | RMSE | {np.mean(ms_valid_rmse_lin):6.3f} +/- {np.std(ms_valid_rmse_lin):6.3f} d^-1")
print(f"Train | R^2  | {np.mean(ms_train_r2_lin):6.3f} +/- {np.std(ms_train_r2_lin):6.3f}")
print(f"Valid | R^2  | {np.mean(ms_valid_r2_lin):6.3f} +/- {np.std(ms_valid_r2_lin):6.3f}")

# Polynomial Reg
model_poly = make_pipeline(StandardScaler(),PolynomialFeatures(include_bias=False),LinearRegression())

# Define the polynomial degrees
param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(1, 5)])

# Hyperparameter optimization using 4-fold cv
model_gscv_poly = GridSearchCV( model_poly,param_grid=param_grid_poly,cv=4,scoring="neg_root_mean_squared_error",return_train_score=True)

# Fitting the model using training data
model_gscv_poly.fit(x_train, y_train)

# Extracted and sorted cv results
df_results_poly = pd.DataFrame(model_gscv_poly.cv_results_)
df_results_poly = df_results_poly.sort_values(by=["rank_test_score"], ascending=True)

# Column names
columns_view_poly = ["rank_test_score","param_polynomialfeatures__degree","mean_test_score","std_test_score","mean_train_score"]

print()
print("Polynomial Regression")
print(df_results_poly.loc[:, columns_view_poly].to_string())

print()
print(f"Best hyperparameters: {model_gscv_poly.best_params_}")
print(f"Best CV RMSE: {-model_gscv_poly.best_score_:.5f} d^-1")


# KNN Reg
model_knn = make_pipeline(StandardScaler(), KNeighborsRegressor())

# Define the grid to search
param_grid = dict(kneighborsregressor__n_neighbors=[i for i in range(1, 81)],kneighborsregressor__weights=["uniform", "distance"])

# Create the model using GridSearchCV
model_gscv = GridSearchCV(model_knn, param_grid=param_grid, cv=4, scoring="neg_root_mean_squared_error", return_train_score=True)

model_gscv.fit(x_train, y_train)

# Extracted and sorted cv results
df_results = pd.DataFrame(model_gscv.cv_results_)
df_results = df_results.sort_values(by=["rank_test_score"], ascending=True)

# Column names
columns_view = ["rank_test_score", "param_kneighborsregressor__n_neighbors", "param_kneighborsregressor__weights", "mean_test_score", "std_test_score", "mean_train_score"]

print()
print("KNN Regression")
#print(df_results.loc[:, columns_view].to_string())

print()
print(f"Best hyperparameters: {model_gscv.best_params_}")
print(f"Best CV RMSE: {-model_gscv.best_score_:.5f} d^-1")

y_test_pred = model_gscv.predict(x_test)
m_test_rmse = root_mean_squared_error(y_test, y_test_pred)
m_test_r2 = r2_score(y_test, y_test_pred)

print()
print(f"Test | RMSE | {m_test_rmse:6.3f} d^-1")
print(f"Test | R^2  | {m_test_r2:6.3f}")

# KNN Validation Curve

# Define figure dimensions
fig_width = 14.39 / 2.54
fig_height = 12.09 / 2.54

# Create the figure
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(fig_width, fig_height),dpi=200, layout="constrained")

# Plotting the RMSE to the correct weighting method
for weight in param_grid["kneighborsregressor__weights"]:

    mask_weight = (df_results.loc[:, "param_kneighborsregressor__weights"] == weight)
    df_sub = df_results.loc[mask_weight, :]

    df_sub = df_sub.sort_values(by=["param_kneighborsregressor__n_neighbors"],ascending=True)

    n_neighbors = df_sub["param_kneighborsregressor__n_neighbors"].to_numpy()

    rmse_train = -df_sub["mean_train_score"].to_numpy()
    rmse_valid = -df_sub["mean_test_score"].to_numpy()

    ax.plot(n_neighbors, rmse_train,label=f"Train ({weight})", linestyle="-")

    ax.plot(n_neighbors, rmse_valid,label=f"Valid ({weight})", linestyle="--")


ax.set_xlabel("Number of Neighbors (k)")
ax.set_ylabel("RMSE [d^-1]")
ax.legend(ncols=1, fontsize=10)
ax.set_ylim(bottom=0)

fig.savefig(os.path.join("figures", "figure_knn_validation.png"))
plt.close(fig)

# KNN vs Predicted Growth Rates

#Create figure
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(fig_width, fig_height),dpi=200, layout="constrained")

# Plot actual vs predicted growth
ax.scatter(y_test, y_test_pred, color="red", marker="o",s=25,alpha=0.7, label="KNN")

y_min = min(0,y_test.min(),y_test_pred.min())
y_max = max(0,y_test.max(),y_test_pred.max())

# Plotting ideal prediction line
ax.plot([y_min, y_max], [y_min, y_max], color="black", linestyle="--", linewidth=2, label="Ideal Prediction")

#Axes
ax.set_xlabel("Actual Specific Growth Rate [d^-1]", fontsize=14)
ax.set_ylabel("Predicted Specific Growth Rate [d^-1]", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=12)

ax.set_xlim(y_min, y_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal", adjustable="box")
ax.legend()


fig.savefig(os.path.join("figures", "figure_03_knn_predictions.png"))
plt.close(fig)

# Models comparison
df_comparison = pd.DataFrame({"Model": ["Linear Regression","Polynomial Regression","KNN Regression"],"CV RMSE [d^-1]": [np.mean(ms_valid_rmse_lin),-model_gscv_poly.best_score_,-model_gscv.best_score_]})

df_comparison = df_comparison.sort_values(by=["CV RMSE [d^-1]"], ascending=True)


print()
print("Model Comparison")
print(df_comparison.to_string(index=False))

# CSV table
df_comparison.to_csv(os.path.join("tables", "table_03_model_comparison.csv"),index=False)

# Final Export Results

# Export polynomial regression results as CSV
df_results_poly.to_csv(os.path.join("tables", "table_04_polynomial_gridsearch.csv"),index=False)

# Export KNN regression hyperparameter results as CSV
df_results.to_csv(os.path.join("tables", "table_02_knn_gridsearch.csv"),index=False)

# CSV summary of the optimized KNN model
df_summary = pd.DataFrame({"Neighbors": [model_gscv.best_params_["kneighborsregressor__n_neighbors"]],"Weights": [model_gscv.best_params_["kneighborsregressor__weights"]],"CV RMSE [d^-1]": [-model_gscv.best_score_],"Test RMSE [d^-1]": [m_test_rmse],"Test R^2": [m_test_r2]})

df_summary.to_csv(os.path.join("tables", "table_01_knn_summary.csv"),index=False)