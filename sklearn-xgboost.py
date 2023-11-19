# XGBoost using sklearn

# Here we pilot a simple pipeline with some preprocessing of predictors
# and a XGBoost model without hyperparameter tuning.
# Some good next steps would be to add more predictors, test different models, tuning, scale the SalePrice column...

# %% import
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

# %% read-data
df_train = pd.read_csv("data/train.csv")

cols_to_keep = ["ExterQual", "LotArea", "YrSold", "SalePrice"]
pred_cols = [x for x in cols_to_keep if x != "SalePrice"]

df = df_train[cols_to_keep]

y = df["SalePrice"]
X = df.drop(["SalePrice"], axis = 1)

# %% Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define which columns should be one-hot encoded and which should be scaled
categorical_cols = ['YrSold', 'ExterQual']
numeric_cols = ['LotArea']

# Preprocessing steps for different column types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_cols),  # MinMaxScaler for numeric columns
        ('cat', OneHotEncoder(), categorical_cols)  # OneHotEncoder for categorical columns
    ])

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Preprocessing step
    ('model', GradientBoostingRegressor())  # XGBoost model
])

# Fit the pipeline (including preprocessing and model) on the training data
pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the test set
score = pipeline.score(X_test, y_test)
print("Pipeline score:", score)

df_test = pd.read_csv("data/test.csv")

df_test.head(5)

test_filter = df_test[pred_cols]

print(test_filter)

yhat_test = pipeline.predict(test_filter)

df_test["SalePrice"] = yhat_test

test_export = df_test[["Id", "SalePrice"]]

test_export.to_csv("data/first_prediction.csv", index=False)

# # Make some plots

train_preds = pipeline.predict(X_train)

df_plot = pd.DataFrame({'yhat': train_preds, 'y': y_train})

import seaborn as sns
import matplotlib.pyplot as plt

# Create a scatterplot
sns.scatterplot(x='yhat', y='y', data=df_plot, color='blue')

# Add labels and title
plt.xlabel('Predicted Values (yhat)')
plt.ylabel('Actual Values (y)')
plt.title('Scatterplot of Predicted vs Actual Values')

# Show the plot
plt.show()

# Calculate residuals
df_plot["residuals"] = df_plot["y"] - df_plot["yhat"]

# Plot residuals
# Create a residual plot using sns.residplot()
sns.residplot(x=df_plot["yhat"], y=df_plot["residuals"], color='blue')

# Add labels and title
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Show the plot
plt.show()

print("Awesome")