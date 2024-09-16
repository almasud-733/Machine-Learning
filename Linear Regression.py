import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1,2,3,4,5]).reshape(-1,1)
Y = np.array([1.2, 1.5, 2.6, 3.2, 3.6])

# create and train LR model
model = LinearRegression()
model.fit(X,Y)

#Predict The sales for 7Th week

week7 =np.array([[12]])
predictLR =model.predict(week7)
print(f"Predict the sales value for 7Th weel: {predictLR}")

#Plotting on Graph
plt.scatter(X,Y, color='blue',label='Sales graph')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.scatter(week7, predictLR, color='green', label='Predicted Sales (Week 7)')
plt.xlabel('Week')
plt.ylabel('Sales')
plt.legend()
plt.title('Linear Regression for Sales Prediction')
plt.show()


