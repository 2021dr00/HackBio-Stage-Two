#Setup & data loading
import numpy as np
import pandas as pd
gdsc = pd.read_excel("C:/Users/Dharshana/Downloads/GDSC.xlsx")
gdsc.head()

#Explore the Dataset
gdsc.shape
gdsc.columns

gdsc["DRUG_NAME"].nunique()
gdsc["TARGET"].nunique()
gdsc["TARGET_PATHWAY"].nunique()

#LN_IC50 is the target variable, while others are features
#Data preprocessing
#Identify missing data
missing = gdsc.isnull().sum().sort_values(ascending=False)
missing.head(20)
#identify the duplicates
gdsc = gdsc.drop_duplicates()
gdsc
#Remove Leakage Variables
leakage_cols = [
    'LN_IC50',
    'AUC',
    'Z_SCORE',
    'COSMIC_ID',
    'DRUG_ID',
    'CELL_LINE_NAME',
    'DRUG_NAME',
    'TCGA_DESC',
    'GDSC Tissue descriptor 1',
    'GDSC Tissue descriptor 2',
    'Screen Medium'
]

#define the features and the target (feature selection and target identification)
X = gdsc.drop(columns=leakage_cols)
y = gdsc["LN_IC50"]

#import libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

#encode the categorical variables in the train set
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
categorical_cols = [
    'Growth Properties',
    'TARGET',
    'TARGET_PATHWAY',
    'Microsatellite instability Status (MSI)',
    'Cancer Type (matching TCGA label)',
    'CNA',
    'Gene Expression',
    'Methylation'
]
preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_cols
        )
    ],
    remainder='passthrough'
)
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

#train a random forest model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)
rf.fit(X_train_encoded, y_train)
#Prediction on the test set
y_pred = rf.predict(X_test_encoded)
y_pred

#evaluation of the model
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
print("R²:", round(r2,3))
print("MAE:", round(mae,3))
print("RMSE:", round(rmse,3))

#Feature Importance
#get the features
feature_names = preprocessor.get_feature_names_out()

print(len(feature_names))

#create feature importance table
importance_gdsc = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
})

importance_gdsc = importance_gdsc.sort_values(
    by='Importance',
    ascending=False
)

print(importance_gdsc.head(20))

#clean the names
importance_gdsc['Feature'] = (
    importance_gdsc['Feature']
    .str.replace('cat__', '', regex=False)
)

# visualise the important features
import seaborn as sns
import matplotlib.pyplot as plt

top20 = importance_gdsc.head(20)

plt.figure(figsize=(10,8))

sns.barplot(
    data=top20,
    x='Importance',
    y='Feature'
)

plt.title("Top 20 Important Features")
plt.show()