import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Relu, Dense

# load the dataset
data = pd.read_csv('Telco_customer_churn.csv')

# check for missing values
data.isnull().sum()

# drop the customerID column as it is not useful for prediction
data.drop('customerID', axis=1, inplace=True)

# convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# split the dataset into features and target variable
X = data.drop('Churn_Yes', axis=1)
y = data['Churn_Yes']

# check the information of the dataset
data.info()

# check the shape of the dataset
data.shape


# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# define a function to evaluate the models
def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


# define the models to be evaluated
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Machine': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
}

# evaluate each model and store the results
results = {}
for name, model in models.items():
    mse, r2 = evaluate_model(model, x_train, y_train, x_test, y_test)
    results[name] = {'MSE': mse, 'R2': r2}

# create a DataFrame to display the results
results_df = pd.DataFrame(results).T

# display the results
print(results_df)

# build a neural network model
nn_model = Sequential()

# add input layer and first hidden layer
nn_model.add(Dense(units=64, activation='relu', input_shape=(x_train.shape[1],)))

# add second hidden layer
nn_model.add(Dense(units=32, activation='relu'))

# add output layer
nn_model.add(Dense(units=1, activation='sigmoid'))

# compile the model
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
history = nn_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# evaluate the model on the test set
test_loss, test_accuracy = nn_model.evaluate(x_test, y_test)

# create a bar plot to compare the performance of the models
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y=results_df['R2'])
plt.title('Model Comparison based on R2 Score')
plt.ylabel('R2 Score')
plt.xlabel('Models')
plt.xticks(rotation=45)
plt.show()

#print all the results
print("Model Results:")
print(results_df)
print(f"Neural Network Test Loss: {test_loss}")
print(f"Neural Network Test Accuracy: {test_accuracy}")
print("Neural Network Training History:")
print(history.history)
print("Neural Network Training Loss over Epochs:")
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Neural Network Training Loss over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()