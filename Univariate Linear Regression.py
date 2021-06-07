import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the csv file
df = pd.read_csv('Simple Linear Regression.csv')

# Compute the x and y coordinates for points on a sine curve
x = df.X
y = df.Y

# Initialise slope and constant
m=1
c=0
alpha = 0.000001


#Minimising Cost Function
for i in range(10):
    temp_1=0
    temp_2=0
    for j in range(45):
        temp_1 += alpha*(m*df.X[j] + c - df.Y[j])*df.X[j]/45
        temp_2 += alpha*(m*df.X[j] + c - df.Y[j])/45
    m -= temp_1
    c -= temp_2
print(m)
print(c)


# Plot the points using matplotlib
plt.scatter(x, y)
plt.plot(x, m*x + c)
plt.show()
