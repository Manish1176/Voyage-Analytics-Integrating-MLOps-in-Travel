import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import numpy as np

# Load the dataset
df = pd.read_excel('flights.xlsx')

print("First 5 rows of the dataset:")
print(df.head(5))

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Prepare features (X) and target (y)
feature_columns = ['from', 'to', 'flightType', 'time', 'distance', 'agency', 'date']
X = df[feature_columns].copy()  # Create a copy to avoid the warning
y = df['price'].copy()

# Convert categorical variables to numerical using Label Encoding
categorical_columns = ['from', 'to', 'flightType', 'agency']
label_encoders = {}

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    X.loc[:, column] = label_encoders[column].fit_transform(X[column])

# Convert date to numerical (assuming it's in datetime format)
X.loc[:, 'date'] = pd.to_datetime(X['date']).astype(np.int64)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
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

# Save the model and encoders
with open('flight_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders
    }, f)

print("\nModel and preprocessing objects saved to 'flight_model.pkl'")