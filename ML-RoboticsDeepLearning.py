import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

# Load the input data
X_train = np.load("robotics_input_data.npy")
y_train = np.load("robotics_output_data.npy")

# Define the model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the data
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Predict the outputs
y_pred = model.predict(X_test)
print("Predicted outputs:", y_pred)
