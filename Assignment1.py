##################################
# 		Machine Learning 	 	 #
# 					 			 #
# 	Simple Linear Regression 	 #
##################################

import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures


# Create a file to write the data to
f = open('data.txt', 'w')

# Create training and testing arrays
training_x 	= np.array([])
testing_x 	= np.array([])

# Generate data from noisy sin function y = sin(x) + e
def calculate_y_values(x):
	# scale=0.1 is a standard deviation of 0.1
	return np.sin(x) + np.random.normal(loc=0, scale=0.1)

# Randomly generate x between 0 and 2pi
for i in range(25):
	training_x = np.append(training_x, random.uniform(0,2*np.pi))
	testing_x = np.append(testing_x, random.uniform(0,2*np.pi))

f.write("Training x values:\n")
f.write(str(training_x))
f.write("\n\nTesting x values:\n")
f.write(str(testing_x))

# Sort for plotting
training_x = np.sort(training_x)
testing_x = np.sort(testing_x)

# Calculate training and testing y values from noisy sin function: y = sin(x) + e
training_y = calculate_y_values(training_x)
testing_y = calculate_y_values(testing_x)

f.write("\n\nTraining y values:\n")
f.write(str(training_y))
f.write("\n\nTesting y values:\n")
f.write(str(testing_y))

# Reshape
training_x = training_x.reshape((-1,1))
training_y = training_y.reshape((-1,1))

# Generate Polynomial Features(cubic)
poly = PolynomialFeatures(3)
features = poly.fit_transform(training_x)

# Transpose and multiply by itself
features_transpose = np.transpose(features)
features_transpose_features = features_transpose.dot(features)

# Get the inverse
inverse = inv(features_transpose_features)

# X transpose Y
x_transpose_y = features_transpose.dot(training_y)

# Combine the 2 sides of the equation to give the 4 beta parameters
beta_parameters = inverse.dot(x_transpose_y)

f.write("\n\nCalculated beta parameters:\n")
f.write(str(beta_parameters))

# Produce estimates y-values
def estimated_y(x, beta_parameters):
	calculated_y = []
	for i in range (len(x)):
		# f(x) = β0 + β1*x + β2*x^2 + β3*x^3
		calculated_y.append(beta_parameters[0] + (beta_parameters[1] * x[i]) + 
			(beta_parameters[2] * (x[i] ** 2)) + (beta_parameters[3] * (x[i] ** 3)))
	return calculated_y

def least_squares(calculated_y, estimated_y):
	matrix = calculated_y - estimated_y
	matrix_transpose = np.transpose(matrix)
	matrix = matrix_transpose.dot(matrix)
	return matrix

# Get the estimated values using the beta parameters
estimated_training_y = estimated_y(training_x, beta_parameters)
estimated_testing_y = estimated_y(testing_x, beta_parameters)

# Write the training and testing calculated values to the data.txt file
f.write("\n\nEstimated training y values:\n")
f.write(str(estimated_training_y))
f.write("\n\nEstimated testing y values:\n")
f.write(str(estimated_testing_y))

# Calculate the training error and testing error
training_error = least_squares(training_y, estimated_training_y)
# Reshape testing_y values first
testing_y = testing_y.reshape((-1,1))
testing_error = least_squares(testing_y, estimated_testing_y)

# Write the training and testing error values to the data.txt file
f.write("\n\nTraining error:\n")
f.write(str(training_error))
f.write("\n\nTesting error:\n")
f.write(str(testing_error))

# Close the file
f.close()

# Plot the training_y points against the resulting function of calculated values
plt.title("Graph of Training Data and Resulting Function")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(training_x, training_y, 'o')
plt.plot(training_x, estimated_training_y, 'r')
plt.show()