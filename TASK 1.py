

# Load the dataset (assuming it's a CSV file)
data = pd.read_csv('house_prices_dataset.csv')

# Data preprocessing
# For simplicity, let's assume preprocessing involves handling missing values and selecting features
# Replace NaN values with mean of the column
data.fillna(data.mean(), inplace=True)

# Select relevant features and target variable
X = data.drop('Price', axis=1)  # Features
y = data['Price']               # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Example of predicting prices for new data
# Assume 'new_data' is a DataFrame containing new data points with the same features as X
# new_data = pd.DataFrame([[...]], columns=X.columns)
# predicted_price = model.predict(new_data)
# print(f'Predicted price: {predicted_price}')
