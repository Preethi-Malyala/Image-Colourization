import pandas as pd
import math
import matplotlib.pyplot as plt


# Read csv File
logic = pd.read_csv('Logistic Regression.csv')


# Define g(z)
def g_of(z):
    return (1+math.exp(-z))**(-1)


# Variables With Feature Scaling
x_1 = (logic.Age - sum(logic.Age)/400)/(max(logic.Age) - min(logic.Age))
x_2 = (logic.EstimatedSalary - sum(logic.EstimatedSalary)/400)/(max(logic.EstimatedSalary) - min(logic.EstimatedSalary))
y = logic.Purchased


# Plotting points
for i in range(400):
    if y[i] == 1:
        plt.scatter(x_1, x_2, y)
for i in range(400):
    if y[i] == 0:
        plt.scatter(x_1, x_2, 1-y)


#Initialise Parameters
theta_0 = 1
theta_1 = 1
theta_2 = 1
alpha = 10
epsilon = 0.01


# Minimise cost function
while True:
    temp_0 = 0
    temp_1 = 0
    temp_2 = 0
    for i in range(400):
        theta_x = theta_0 + theta_1 * x_1[i] + theta_2 * x_2[i]
        temp_0 += (g_of(theta_x) - y[i])
        temp_1 += (g_of(theta_x) - y[i]) * x_1[i]
        temp_2 += (g_of(theta_x) - y[i]) * x_2[i]

    theta_0 -= temp_0*alpha/400
    theta_1 -= temp_1*alpha/400
    theta_2 -= temp_2*alpha/400

    print(str(theta_0) + ' + ' + str(theta_1) + '*x_1' +' + ' + str(theta_2) + '*x_2')


    if abs(temp_0)<epsilon and abs(temp_1)<epsilon and abs(temp_2)<epsilon :
        break


# Print final parameters
print('Hypothesis = ' + str(theta_0) + ' + ' + str(theta_1) + '*x_1' +' + ' + str(theta_2) + '*x_2')


# Plotting 2.0
plt.plot(x_1, (-theta_0 - theta_1*x_1)/theta_2)
plt.show()