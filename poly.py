'''
Polynomial regression is a special case of multiple linear regression that adds terms with degrees
greater than one to the model. Polynomial regression is used to find a relationship between two variables.
In statistics, polynomial regression is a form of regression analysis in which the relationship between the
independent variable x and the dependent variable y is modelled as an nth degree polynomial in x.
Polynomial regression fits a nonlinear relationship between the value of x and the corresponding conditional mean of y
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
X_train = [[6], [8], [10], [14], [18]]  # diamters of pizzas
y_train = [[7], [9], [13], [17.5], [18]]  #prices of pizzas

#Testing set
X_test = [[6], [8], [11], [16]]  # diamters of pizzas
y_test = [[8], [12], [15], [18]]  # prices of pizzas

#Creating subplots
figure, axes = plt.subplots(1, 2, figsize=(20, 8))

#Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
axes[0].plot(xx, yy)

#Trying to find relation of different degrees used to create Polynomial degrees
#Creating train and test accuracy lists to store accuracy of training and testing data on various polynomial models
train_acc = []
test_acc = []
for degree in range(2, 12, 2):
    #Set the degree of the Polynomial Regression model
    poly_featurizer = PolynomialFeatures(degree=degree)

    #This preprocessor transforms an input data matrix into a new data matrix of a given degree
    X_train_poly = poly_featurizer.fit_transform(X_train)
    X_test_poly = poly_featurizer.transform(X_test)

    #Train and test the regressor_quadratic model
    regressor_poly = LinearRegression()
    regressor_poly = regressor_poly.fit(X_train_poly, y_train)
    xx_poly = poly_featurizer.transform(xx.reshape(xx.shape[0], 1))

    #Calculating accuracy for training and testing data and storing in list
    train_score = regressor_poly.score(X_train_poly, y_train)
    train_acc.append(train_score)
    test_score = regressor_poly.score(X_test_poly, y_test)
    test_acc.append(test_score)
    #Plot the graph
    axes[0].plot(xx, regressor_poly.predict(xx_poly), linestyle='--', label='Model on degree: {}'.format(degree))

#labeling
axes[0].legend()
axes[0].set_title('Pizza price regressed on diameter')
axes[0].set_xlabel('Diameter in inches')
axes[0].set_ylabel('Price in dollars')
axes[0].axis([0, 25, 0, 25])
axes[0].grid(True)
#Scatter plot for plotting training points
axes[0].scatter(X_train, y_train)

#Ploting for test and train accuracy
axes[1].plot(range(2, 12, 2), train_acc, label='Training Accuracy')
axes[1].plot(range(2, 12, 2), test_acc, label='Training Accuracy')
axes[1].set_xlabel('Polynomial Degree')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and testing set accuracy w.r.t polynomial model degrees')
axes[1].legend()
axes[1].grid(True)
plt.show()
figure.savefig('results.png', bbox_inches='tight')

'''Figure 1: As the degree of polynomial increase the model overfitted the training set
Meaning the production of an analysis corresponds too closely or exactly to a particular set of data, 
and may therefore fail to fit additional data or predict future observations
'''

'''Figure 2: As the degree of polynomial increased, training set accuracy increased but 
testing set accuracy decreased causing both variance and bias to increase in the model because of overfitting.
'''

'''Hence, according to lower complexity of data, we can say degree order = 2 polynomial regressor worked best 
on dataset and made the model more generalised on data compared to others.
'''

'''The main concept to find relationships using Polynomial regression is to find the relationship between different 
degrees of polynomial regressor using accuracy, variance, and bias.
'''
