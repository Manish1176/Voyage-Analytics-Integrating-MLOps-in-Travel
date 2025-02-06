import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import numpy as np
import xgboost as xgb

#load data
df = pd.read_excel('users.xlsx')

print("First 5 rows of the dataframe: ")
print(df.head(5))

#Check for null values
print("Null values in the dataframe: ")
print(df.isnull().sum())

#Drop null values
df = df.dropna()    

#prepare features and target
feature_columns = ['userCode','company','name','age']
X = df[feature_columns].copy()
y = df['gender'].copy()

# Print unique gender values to check what we're working with
print("\nUnique gender values before mapping:")
print(y.value_counts(dropna=False))

# Clean gender values (strip whitespace and convert to title case)
y = y.str.strip().str.title()

# Check for any invalid gender values
valid_genders = {'Male', 'Female'}
invalid_genders = set(y.unique()) - valid_genders
if invalid_genders:
    print(f"\nWarning: Found invalid gender values: {invalid_genders}")
    
#Convert target to binary
y = y.map({'Male': 0, 'Female': 1})

# Check if we have any NaN values after mapping
if y.isna().any():
    print("\nWarning: NaN values found after gender mapping!")
    print("Rows with NaN gender values:")
    print(df[y.isna()][['userCode', 'gender']])
    # Drop rows with NaN gender values
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    print(f"\nRemoved {(~valid_indices).sum()} rows with invalid gender values")

#Convert categorical variables to numeric
categorical_columns = ['company','name']
label_encoders = {}

for column in categorical_columns:
    label_encoder = LabelEncoder()
    X[column] = label_encoder.fit_transform(X[column])
    label_encoders[column] = label_encoder


#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train model
model = xgb.XGBClassifier(random_state=42)
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print("\nModel Performance Metrics:")
print(f"R-squared Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")  

#Save model
with open('gender_classification_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders
    }, f)

print("\nModel and preprocessing objects saved to 'gender_classification_model.pkl'")