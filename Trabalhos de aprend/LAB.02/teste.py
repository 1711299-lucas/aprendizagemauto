import pandas as pd
from sklearn import linear_model
import pickle as p1

# Load the dataset
data = pd.read_csv("optdigits.tes", delim_whitespace=True, header=None)

# Split the data into training features and target variable
train_data = data[:3133]
data_X = train_data.iloc[:, 1:8]
data_Y = train_data.iloc[:, 8]  # Using Series for compatibility

# Create and fit the linear regression model
regr = linear_model.LinearRegression()
preditor_linear_model = regr.fit(data_X, data_Y)

# Save the model to a file
with open('optical+recognition+of+handwritten+digits.zip_model.pkl', 'wb') as preditor_Pickle:  # Using context manager for file handling
    p1.dump(preditor_linear_model, preditor_Pickle)

print("Model saved as optdigits_model.pkl")
