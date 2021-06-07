import pandas as pd


#Read the csv file
cars = pd.read_csv('Multivariate Linear Regression.csv')


# Initialise parameters
theta_0 = 1
theta_1 = 1
theta_2 = 1
alpha = 0.1
epsilon = 0.01


# Variables With Feature Scaling
x_1 = (cars.Volume - sum(cars.Volume)/36)/(max(cars.Volume) - min(cars.Volume))
x_2 = (cars.Weight - sum(cars.Weight)/36)/(max(cars.Weight) - min(cars.Weight))
y = (cars.CO2 - sum(cars.CO2)/36)/(max(cars.CO2) - min(cars.CO2))


# Minimising Cost Function
while True:
    temp_0 = sum(theta_0 + theta_1*x_1 + theta_2*x_2 - y)
    temp_1 = sum((theta_0 + theta_1*x_1 + theta_2*x_2 - y)*x_1)
    temp_2 = sum((theta_0 + theta_1*x_1 + theta_2*x_2 - y)*x_2)

    theta_0 -= temp_0*alpha/36
    theta_1 -= temp_1*alpha/36
    theta_2 -= temp_2*alpha/36
    # Defining Cost function
    cost = sum((theta_0 + theta_1*x_1 + theta_2*x_2 - y)**2)/72

    print(cost)
    if abs(temp_0)<epsilon and abs(temp_1)<epsilon and abs(temp_2)<epsilon:
        break


# Print final parameters
print('Hypothesis = ' + str(theta_0) + ' + ' + str(theta_1) + '*x_1' +' + ' + str(theta_2) + '*x_2')