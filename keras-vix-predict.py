#from: https://github.com/kimanalytics/Recurrent-Neural-Network-to-Predict-Stock-Prices/blob/master/Recurrent%20Neural%20Network%20to%20Predict%20Tesla%20Stock%20Prices.ipynb
#to upload local file to juypter simply hit upload in main directory
#block comment is ctrl+/ after selecting lines of code

#Block 1: data visualization
#1)
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading CSV file into training set
training_set = pd.read_csv('vix-daily-data.csv')
training_set.head()

#2)
# Reading CSV file into test set
test_set = pd.read_csv('vix-1yr-data-092017-092018.csv')
test_set.head()

#Block 2:data processing
#3)
# Getting relevant feature from training set(close price)
training_set = training_set.iloc[:,4:6] #isolate 1st and close price columns
training_set.head()

#4)
# Converting to 2D array
training_set = training_set.values
training_set

#5)
# Feature Scaling using normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
training_set

#6)
# Getting the inputs and the outputs(3706 data points)
#X_train = training_set[0:3706] #input
X_train = training_set[1:3706] #input
y_train = training_set[1:3707] #output, shift to 1 day in future
#length in both arrays must match but y_train is 1 day in future so take actual data points and -1(1983 -> 1982 points)
#X_train = training_set[496:2478] #randomly chosen, 1982
#y_train = training_set[1724:3707] #randomly chosen, 1983 but last point doesnt exist

# Example
today = pd.DataFrame(X_train[0:5])
tomorrow = pd.DataFrame(y_train[0:5])
ex = pd.concat([today, tomorrow], axis=1)
ex.columns = (['today', 'tomorrow'])
ex

#7)
# Reshaping into required shape for Keras processing
#(args: batch_size, timesteps, input_dim) = datapoints, time interval, datapoint per interval
#X_train = np.reshape(X_train, (3706, 1, 1))
X_train = np.reshape(X_train, (3705, 1, 1))
#X_train = np.reshape(X_train, (1982, 1, 1)) #see step 6 for reason, take array length and minus 1
X_train

#Block 3:building neural net
#8)
# Importing the Keras libraries and packages
from keras.models import Sequential #neural net in consecutive order
from keras.layers import Dense #for the nodes
from keras.layers import LSTM #type of model

#9)
# Initializing the Recurrent Neural Network
regressor = Sequential() #this is regression, not classification

#10)
# Adding the input layer and the LSTM layer
#none because no idea input shape, 1 because each day has 1 datapoint
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

#11)
# Adding the output layer
regressor.add(Dense(units = 1))

#12)
# Compiling the Recurrent Neural Network
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#13)
# Fitting the Recurrent Neural Network to the Training set
regressor.fit(X_train, y_train, epochs = 200, batch_size = 32)

#Block 4:Make prediction and visualization
#14)
# Getting the real vix prices
test_set = pd.read_csv('vix-1yr-data-092017-092018.csv')
test_set.head()

#15)
# Getting relevant feature
real_vix_value = test_set.iloc[:,4:6]
real_vix_value.head()

#16)
# Converting to 2D array
real_vix_value = real_vix_value.values

#17)
# Getting the predicted current vix value
inputs = real_vix_value
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (253, 1, 1)) #253 datapoints in csv
#inputs = np.reshape(inputs, (3706, 1, 1))
predicted_vix_value = regressor.predict(inputs)
predicted_vix_value = sc.inverse_transform(predicted_vix_value)

#18)
# Visualizing the results
plt.plot(real_vix_value, color = 'red', label = 'Real vix value')
plt.plot(predicted_vix_value, color = 'blue', label = 'Predicted vix value')
plt.title('Vix value Prediction')
plt.xlabel('Days')
plt.ylabel('Vix value')
plt.legend()
plt.show()

#19)
# Getting the real vix value of past 5 years
real_vix_value_train = pd.read_csv('vix-daily-data.csv')
real_vix_value_train = real_vix_value_train.iloc[:,4:6].values

# Getting the predicted vix value of past 5 years
predicted_vix_value_train = regressor.predict(X_train)
predicted_vix_value_train = sc.inverse_transform(predicted_vix_value_train)

# Visualising the results
plt.plot(real_vix_value_train, color = 'red', label = 'Real vix value')
plt.plot(predicted_vix_value_train, color = 'blue', label = 'Predicted vix value')
plt.title('5-year vix value Prediction')
plt.xlabel('Days')
plt.ylabel('vix value')
plt.legend()
plt.show()

#20)
#visualize real data
plt.plot(real_vix_value_train, color = 'red', label = 'Real vix value')
plt.title('5-year vix value')
plt.xlabel('Days')
plt.ylabel('vix value')
plt.legend()
plt.show()

#21)
#visualize predicted data
plt.plot(predicted_vix_value_train, color = 'blue', label = 'Predicted vix value')
plt.title('5-year vix value Prediction')
plt.xlabel('Days')
plt.ylabel('vix value')
plt.legend()
plt.show()